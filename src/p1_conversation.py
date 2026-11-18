"""Part 1 conversation memory (Kavan).

One Conversation object per conversation_id. It keeps:
- the earlier turns, so follow-ups like "And in 2013?" can be rewritten as full questions,
- the values the user supplied ("assume...", "for this chat..."), as structured session facts.

Session facts live only in this object. They are never written to the index, and they are
gone when the object is dropped at the end of the conversation. Requests to change the
documents permanently are refused, because the knowledge base is read-only.
"""
from __future__ import annotations

import re

from p1_engine import (ENTITIES, NOUNS, VARIABLES, SessionFact, Turn, entities_in_order, find_entity, find_variable,
                       labels_in, nearest_term, resolve, variables_in_order)

SESSION_RE = re.compile(
    r"\b(assume|assuming|suppose|supposing|let'?s say|pretend|imagine|hypothetical(?:ly)?|what if|take it as given"
    r"|for (?:this|our) (?:chat|conversation|session)|(?:in|within) (?:this|our) (?:chat|conversation|session)"
    r"|(?:this|our) (?:chat|conversation|session) only|new data point)\b", re.I)
CHANGE_VERBS = re.compile(r"\b(update|change|edit|modify|overwrite|rewrite|save|store|write|add|insert|correct|fix|replace|delete|"
                          r"remove|amend|record)\b", re.I)
PERMANENT_WORDS = re.compile(r"\b(permanent(?:ly)?|knowledge base|database|vector (?:store|db|database)|the index|for good|"
                             r"for everyone|(?:all|future) (?:future )?(?:users|conversations|chats|sessions)|"
                             r"the (?:documents?|pdfs?|sources?|files?) (?:so (?:that )?|to )?(?:it |they )?(?:says?|reads?|shows?))\b", re.I)
PRESENCE_RE = re.compile(r"\b(?:is|are|was|were)\b[^?]*\b(?:in|from|inside)\s+(?:your|the|these|those)\s+"
                         r"(?:documents?|docs|knowledge base|sources|pdfs?|files)\b"
                         r"|\bdo(?:es)? (?:the|your) (?:documents?|sources) (?:report|contain|include|mention|have)\b", re.I)
YES_NO_START = re.compile(r"\s*(is|are|was|were|do|does|did)\b", re.I)
FOLLOW_START = re.compile(r"^\s*(and|what about|how about|also|then|same|in)\b", re.I)
THAT_PERIOD = re.compile(r"\b(?:in\s+)?(?:that|the same|this)\s+(?:year|period)\b", re.I)
POINTER = re.compile(r"\b(that|this|it|its|those|these|the same)\b", re.I)
NUMBER = re.compile(r"(?<![\w.])([+\-−]?\d+(?:\.\d+)?)\s*%")
FALL_BEFORE = re.compile(r"\b(fell|fall|falls|declined?|contracted?|shrank|shrunk|shrink|dropped?|decreased?)\s+(?:by\s+)?$", re.I)


class Conversation:
    def __init__(self, conversation_id: str, engine, mode: str = "hybrid"):
        self.conversation_id, self.engine, self.mode = conversation_id, engine, mode
        self.history: list[Turn] = []
        self.session_facts: list[SessionFact] = []
        self.context = {"entity": None, "variable": None, "period": None}
        self.last_standalone = ""

    # ---- the main call
    def answer(self, question: str) -> dict:
        standalone = question
        if self._is_permanent_change(question):
            out = self._refuse(question)
        else:
            new_fact = self._read_session_value(question)
            presence = self._presence_answer(question) if self.session_facts else None
            if presence is not None:
                out = presence
            else:
                standalone = self._standalone(question) if self.history else question
                out = self.engine.answer(standalone, self.history, self.session_facts, self.mode)
                if new_fact is not None and "?" not in question:
                    out = self._acknowledge(new_fact)
        self._update_context(standalone)
        self.history.append(Turn(question, standalone, out))
        self.last_standalone = standalone
        return out

    # ---- follow-ups
    def _update_context(self, text: str) -> None:
        ents, variables = entities_in_order(text), variables_in_order(text)
        periods = self.engine._periods(text)
        if ents:
            self.context["entity"] = ents[0]
        if variables:
            self.context["variable"] = variables[0]
        if periods:
            self.context["period"] = periods[-1]

    def _standalone(self, question: str) -> str:
        """Rewrite a follow-up as a full question, using only this conversation's earlier turns."""
        ctx, prev = self.context, self.history[-1].standalone_question
        q = question.strip()
        ents, variables, labels = find_entity(q), find_variable(q), labels_in(q)
        periods = self.engine._periods(q)
        if ctx["period"] and THAT_PERIOD.search(q):
            q = THAT_PERIOD.sub(f"in {ctx['period']}", q)
            periods = [ctx["period"]]
        looks_like_follow_up = (FOLLOW_START.search(question) or THAT_PERIOD.search(question) or POINTER.search(question)
                                or SESSION_RE.search(question) or len(question.split()) <= 8)
        if labels or not looks_like_follow_up:
            return q
        prev_periods = self.engine._periods(prev)
        # "And in 2013?": the previous question with the new year
        if not ents and not variables and len(periods) == 1 and len(prev_periods) == 1 and len(question.split()) <= 6:
            return re.sub(rf"\b{re.escape(prev_periods[0])}\b", periods[0], prev)
        # "What about China?": the previous question with the new entity
        if ents and not variables and not periods and len(question.split()) <= 6 and find_entity(prev):
            new = prev
            for pat in ENTITIES[find_entity(prev)[0]]:
                new = re.sub(pat, ents[0], new, flags=re.I)
            return new
        missing = []
        if not ents and ctx["entity"]:
            missing.append(ctx["entity"])
        if not variables and ctx["variable"]:
            missing.append(ctx["variable"])
        if not periods and ctx["period"] and FOLLOW_START.search(question):
            missing.append(ctx["period"])
        if missing:
            q = q.rstrip(" ?") + f" ({', '.join(missing)})?"
        return q

    # ---- user-supplied values
    def _read_session_value(self, question: str) -> SessionFact | None:
        """Store "assume X was 6.0% in 2009" style values as a structured fact in this conversation only."""
        if not SESSION_RE.search(question):
            return None
        ctx = self.context
        for sent in re.split(r"(?<!U\.S\.)(?<=[.!?:])\s+", question):  # "U.S." does not end a sentence
            m = NUMBER.search(sent)
            if not m:
                continue
            window = sent[:m.start()]
            entity = nearest_term(window, ENTITIES) or (find_entity(sent) or [ctx["entity"]])[0]
            variable = nearest_term(window, VARIABLES) or (find_variable(sent) or [ctx["variable"]])[0]
            periods = self.engine._periods(sent)
            period = periods[0] if periods else ctx["period"]
            raw = m.group(1).replace("−", "-")
            value = float(raw)
            if not raw.startswith(("+", "-")) and FALL_BEFORE.search(window[-30:]):
                value = -value
            fact = SessionFact(sent.strip(), len(self.history) + 1, entity, variable, period, value)
            if entity is None or variable is None or period is None:
                fact.entity = fact.variable = fact.period = fact.value = None  # keep the text, the model still sees it
            self.session_facts = [f for f in self.session_facts
                                  if fact.entity is None or (f.entity, f.variable, f.period) != (fact.entity, fact.variable, fact.period)]
            self.session_facts.append(fact)
            if entity:
                self.context.update(entity=entity, variable=variable or ctx["variable"], period=period or ctx["period"])
            return fact
        return None

    def _doc_value(self, fact: SessionFact):
        return resolve(self.engine.lookup(fact.entity, fact.variable, fact.period))

    def _describe(self, fact: SessionFact) -> tuple[str, list[str], list[str]]:
        """Sentence about what the documents report for the user's value, plus citations."""
        noun = NOUNS.get(fact.variable, fact.variable)
        res = self._doc_value(fact)
        if res is None or res.value is None:
            known = [f for f in self.engine.kb.facts if (f.entity, f.variable) == (fact.entity, fact.variable) and f.period.isdigit()]
            if known:
                last = max(known, key=lambda f: (int(f.period), f.kind != "forecast"))
                latest = resolve(self.engine.lookup(last.entity, last.variable, last.period))
                citations, refs = self.engine._cite(latest.chosen)
                return (f"The documents do not report {fact.entity} {noun} for {fact.period}; their latest value is "
                        f"{latest.value:g}% for {last.period} ({self.engine._where(latest.chosen[0])}).", citations, refs)
            return f"The documents do not report {fact.entity} {noun} for {fact.period}.", [], []
        citations, refs = self.engine._cite(res.chosen)
        return f"The documents report {res.value:g}% for {fact.entity} {noun} in {fact.period} ({self.engine._where(res.chosen[0])}).", citations, refs

    def _acknowledge(self, fact: SessionFact) -> dict:
        if fact.entity is None:
            return {"answer": "Noted for this conversation only. The knowledge base is unchanged.", "status": "session_override",
                    "citations": [], "figure_references": []}
        noun = NOUNS.get(fact.variable, fact.variable)
        doc, citations, refs = self._describe(fact)
        return {"answer": f"Under your assumption for this conversation, {fact.entity} {noun} in {fact.period} is {fact.value:g}%. {doc} "
                          "The knowledge base is unchanged.", "status": "session_override", "citations": citations, "figure_references": refs}

    def _presence_answer(self, question: str) -> dict | None:
        """'Is that figure in your documents?' about a value the user gave us."""
        if not (YES_NO_START.match(question) and PRESENCE_RE.search(question)):
            return None
        structured = [f for f in self.session_facts if f.entity is not None]
        if not structured:
            return None
        periods = self.engine._periods(question)
        fact = next((f for f in reversed(structured) if f.period in periods), None) if periods else None
        if fact is None:
            if not POINTER.search(question):
                return None
            fact = structured[-1]
        doc, citations, refs = self._describe(fact)
        res = self._doc_value(fact)
        noun = NOUNS.get(fact.variable, fact.variable)
        if res is None or res.value is None:
            return {"answer": f"No. The {fact.period} value of {fact.value:g}% for {fact.entity} {noun} came from you in this conversation. "
                              f"{doc}", "status": "not_found", "citations": citations, "figure_references": refs}
        return {"answer": f"Yes, the documents cover it, but with a different value from yours. {doc}", "status": "answered",
                "citations": citations, "figure_references": refs}

    # ---- permanent changes
    def _is_permanent_change(self, question: str) -> bool:
        return bool(CHANGE_VERBS.search(question) and PERMANENT_WORDS.search(question))

    def _refuse(self, question: str) -> dict:
        text = ("I can't do that. The documents are a read-only knowledge base, so they cannot be changed permanently.")
        structured = [f for f in self.session_facts if f.entity is not None]
        if not structured:
            return {"answer": text + " The knowledge base is unchanged.", "status": "answered", "citations": [], "figure_references": []}
        fact = structured[-1]
        noun = NOUNS.get(fact.variable, fact.variable)
        doc, citations, refs = self._describe(fact)
        return {"answer": f"{text} Your value of {fact.value:g}% for {fact.entity} {noun} in {fact.period} applies only in this "
                          f"conversation. {doc} The knowledge base is unchanged.", "status": "session_override",
                "citations": citations, "figure_references": refs}
