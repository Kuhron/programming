import sys
import random
import json
import string
import os



class Exam:
    def __init__(self, questions_by_type):
        self.questions_by_type = questions_by_type

    @staticmethod
    def from_json(fp):
        with open(fp) as f:
            d = json.load(f)
        assert set(d.keys()) == {"questions"}
        questions_by_type = {}
        for qd in d["questions"]:
            q = Question.from_object(qd)
            qtype = type(q).qtype
            if qtype not in questions_by_type:
                questions_by_type[qtype] = []
            questions_by_type[qtype].append(q)
        return Exam(questions_by_type)

    def get_new_version(self, codename):
        new_questions_by_type = {}
        for qtype in Question.types:
            if qtype not in self.questions_by_type:
                continue
            questions_of_type = [x for x in self.questions_by_type[qtype]]
            random.shuffle(questions_of_type)
            new_questions_by_type[qtype] = []
            for q in questions_of_type:
                answers = q.get_answer_choices()
                new_q = QuestionToPrint(q.prompt, answers)
                new_questions_by_type[qtype].append(new_q)
        return ExamVersion(codename, new_questions_by_type)


class ExamVersion:
    def __init__(self, codename, questions_by_type):
        self.questions_by_type = questions_by_type

    def print(self, file=None):
        fprint = lambda *args, **kwargs: print(*args, **kwargs, file=file)
        for qtype_i, qtype in enumerate(x for x in Question.types if x in self.questions_by_type):
            section_letter = string.ascii_uppercase[qtype_i]
            questions_of_type = self.questions_by_type[qtype]
            section_title = Question.qtype_to_section_title(qtype)
            fprint(f"<h3>Section {section_letter}: {section_title}</h3>")
            fprint()
            fprint("<ul>")
            for q_i, q in enumerate(questions_of_type):
                assert type(q) is QuestionToPrint  # not the actual question types
                q.print(label=f"{section_letter}{q_i+1}", file=file)
            fprint("</ul>")
            fprint("----")


class QuestionToPrint:
    def __init__(self, prompt, answer_choices):
        self.prompt = prompt
        self.answer_choices = answer_choices

    def print(self, label=None, file=None):
        fprint = lambda *args, **kwargs: print(*args, **kwargs, file=file)
        prompt_str = (f"{label}. " if label is not None else "") + self.prompt
        fprint(prompt_str)
        if self.answer_choices is None:
            fprint("<p>______</p>")
        else:
            fprint("<ol type='a'>")
            for a in self.answer_choices:
                fprint(f"<li>{a}</li>")
            fprint("</ol>")


class MultipleChoiceQuestion:
    qtype = "multiple-choice"
    section_title = "Multiple Choice"

    def __init__(self, prompt, answer, distractors):
        self.prompt = prompt
        self.answer = answer
        self.distractors = distractors
        assert len(distractors) > 0, "need some distractors for multiple choice questions"
        assert answer not in distractors, "answer cannot be in distractors"

    @staticmethod
    def from_object(d):
        assert d["type"] == MultipleChoiceQuestion.qtype
        prompt = d["prompt"]
        answer = d["answer"]
        distractors = d["distractors"]
        return MultipleChoiceQuestion(prompt, answer, distractors)

    def get_answer_choices(self, shuffle=True):
        lst = [self.answer] + self.distractors
        if shuffle:
            random.shuffle(lst)
        return lst


class TrueOrFalseQuestion:
    qtype = "true-or-false"
    section_title = "True/False"

    def __init__(self, prompt, answer):
        self.prompt = prompt
        if answer == "true":
            self.answer = answer
            self.distractors = ["false"]
        elif answer == "false":
            self.answer = answer
            self.distractors = ["true"]
        else:
            raise ValueError(f"bad answer for true/false question: {answer}")

    @staticmethod
    def from_object(d):
        assert d["type"] == TrueOrFalseQuestion.qtype
        prompt = d["prompt"]
        answer = d["answer"]
        return TrueOrFalseQuestion(prompt, answer)

    def get_answer_choices(self, shuffle=False):
        if shuffle:
            raise ValueError("don't shuffle answers for true/false")
        return ["true", "false"]


class ShortAnswerQuestion:
    qtype = "short-answer"
    section_title = "Short Answer"

    def __init__(self, prompt):
        self.prompt = prompt

    @staticmethod
    def from_object(d):
        assert d["type"] == ShortAnswerQuestion.qtype
        prompt = d["prompt"]
        return ShortAnswerQuestion(prompt)

    def get_answer_choices(self):
        return None


class Question:
    types = {cls.qtype : cls for cls in [
        MultipleChoiceQuestion,
        TrueOrFalseQuestion,
        ShortAnswerQuestion,
    ]}

    @staticmethod
    def from_object(d):
        qtype = d["type"]
        typ = Question.types[qtype]
        return typ.from_object(d)

    @staticmethod
    def qtype_to_section_title(qtype):
        cls = Question.types[qtype]
        return cls.section_title



if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        inp_fp = args[0]
        out_dir = args[1]
        n_versions = int(args[2])
    except (ValueError, IndexError):
        print("usage: python TestShuffler.py [INPUT_JSON_FP:str] [OUTPUT_DIR:str] [N_EXAM_VERSIONS:int]")
        sys.exit()

    exam = Exam.from_json(inp_fp)

    words = [
        "jackalope",
        "sasquatch",
        "macadamia",
        "dachshund",
    ]
    version_names = random.sample(words, n_versions)
    versions = [exam.get_new_version(version_names[i]) for i in range(n_versions)]
    for i, v in enumerate(versions):
        fp = os.path.join(out_dir, f"ExamVersion{i+1}.html")
        with open(fp, "w") as f:
            print(f"\n---- version {i+1}, codename {version_names[i]} ----", file=f)
            v.print(file=f)

