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
        answers = {}
        for qtype in Question.types:
            if qtype not in self.questions_by_type:
                continue
            questions_of_type = [x for x in self.questions_by_type[qtype]]

            if False: #qtype == "transcription-and-parsing":
                # hack to keep the parsing question at the end
                assert len(questions_of_type) == 3
                questions_of_type = (questions_of_type[:2])[::random.choice([-1, 1])] + [questions_of_type[-1]]
            else:
                random.shuffle(questions_of_type)

            new_questions_by_type[qtype] = []
            answers[qtype] = []
            for q in questions_of_type:
                answer_choices = q.get_answer_choices()
                answer = getattr(q, "answer", None)

                if qtype == "transcription-and-parsing":
                    items, answers_by_item = q.get_items_and_answers()
                else:
                    items, answers_by_item = None, None
                
                assert not (answer is not None and answers_by_item is not None), "question cannot have a single answer as well as answers for each of its items"
                if answer is None:
                    answer = answers_by_item

                image_fp = getattr(q, "image_fp", None)
                new_q = QuestionToPrint(prompt=q.prompt, answer=answer, answer_choices=answer_choices, items=items, image_fp=image_fp)
                new_questions_by_type[qtype].append(new_q)

                answers[qtype].append(answer)
        return ExamVersion(codename, new_questions_by_type, answers)


class ExamVersion:
    def __init__(self, codename, questions_by_type, answers):
        self.codename = codename
        self.questions_by_type = questions_by_type
        self.answers = answers

    def print(self, file=None, with_answers=False):
        fprint = lambda *args, **kwargs: print(*args, **kwargs, file=file)

        with open("boilerplate.html") as f:
            boilerplate = f.read().strip()
        boilerplate = boilerplate.replace("<%% codename %%>", f"<p class=\"columns right-aligned\">" + " "*40 + f"{self.codename}</p>")
        fprint(boilerplate)

        sections_to_put_on_new_pages = ["transcription-and-parsing", "extra-credit-with-image"]

        qtypes = [x for x in Question.types if x in self.questions_by_type]
        for qtype_i, qtype in enumerate(qtypes):
            section_letter = string.ascii_uppercase[qtype_i]
            questions_of_type = self.questions_by_type[qtype]
            section_title = Question.qtype_to_section_title(qtype)
            if qtype in sections_to_put_on_new_pages:
                fprint("<div class=\"pagebreak\"> </div>")
            fprint(f"<h3>Section {section_letter}: {section_title}</h3>")
            fprint()
            for q_i, q in enumerate(questions_of_type):
                assert type(q) is QuestionToPrint  # not the actual question types
                # q.print(label=f"{section_letter}{q_i+1}", file=file)
                q.print(label=f"{q_i+1}", file=file, with_answers=with_answers)
                fprint("<br>")
        
        with open("boilerplate_end.html") as f:
            boilerplate = f.read().strip()
        fprint(boilerplate)


class QuestionToPrint:
    def __init__(self, *, prompt, answer, answer_choices=None, items=None, image_fp=None):
        self.prompt = prompt
        self.answer = answer
        self.answer_choices = answer_choices
        self.items = items
        self.image_fp = image_fp

    def print(self, label=None, file=None, with_answers=False):
        p = QuestionToPrint.get_answer_choice_string
        fprint = lambda *args, **kwargs: print(*args, **kwargs, file=file)
        fprint("<div>")
        prompt_str = "<p>" + (f"{label}. " if label is not None else "") + self.prompt + "</p>"
        fprint(prompt_str)
        if self.image_fp is not None:
            fprint(f"<br><img src=\"{self.image_fp}\"/><br><br>")
        if self.answer_choices is None and self.items is None:
            # short answer
            n_breaks = 5
            if with_answers:
                a = self.answer
                fprint(f"<p><b>ANSWER: {p(a)}</b></p>" + "<br>"*2)
            else:
                fprint("<br>" * n_breaks)
        else:
            if self.answer_choices is not None:
                fprint("<ol type='a'>")
                for a in self.answer_choices:
                    if with_answers and a == self.answer:
                        fprint(f"<li><b>{p(a)}</b> &lt;-- THIS IS THE ANSWER</li>")
                    else:
                        fprint(f"<li>{p(a)}</li>")
                fprint("</ol>")
            if self.items is not None:
                fprint("<ul>")
                for item_i, item in enumerate(self.items):
                    fprint(f"<li>{item}</li>")
                    if with_answers:
                        a = self.answer[item_i]
                        fprint(f"<ul><li><b>ANSWER: {p(a)}</b></li></ul>")
                    else:
                        fprint("<br>")
                    fprint("<br><br>")
                fprint("</ul>")
        fprint("</div>")
    
    def get_answer_choice_string(s):
        if s == "<aabove>":
            return "all of the above"
        elif s == "<nabove>":
            return "none of the above"
        else:
            return s


class MultipleChoiceQuestion:
    qtype = "multiple-choice"
    section_title = "Multiple Choice"

    def __init__(self, prompt, answer, distractors):
        self.prompt = prompt
        self.answer = answer
        self.distractors = distractors
        assert len(distractors) > 0, "need some distractors for multiple choice questions"
        assert answer not in distractors, "answer cannot be in distractors"

    @classmethod
    def from_object(cls, d):
        assert d["type"] == cls.qtype
        prompt = d["prompt"]
        answer = d["answer"]
        distractors = d["distractors"]
        return cls(prompt, answer, distractors)

    def get_answer_choices(self, shuffle=True):
        aabove = "<aabove>"  # all of the above
        nabove = "<nabove>"  # none of the above
        lst = [self.answer] + self.distractors
        if shuffle:
            ans_lst = [x for x in lst if x not in [aabove, nabove]]
            random.shuffle(ans_lst)
            if aabove in lst:
                ans_lst.append(aabove)
            if nabove in lst:
                ans_lst.append(nabove)
        else:
            ans_lst = [x for x in lst]
        return ans_lst


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
            raise ValueError("don't shuffle answer choices for true/false")
        return ["true", "false"]


class ShortAnswerQuestion:
    qtype = "short-answer"
    section_title = "Short Answer"

    def __init__(self, prompt, answer):
        self.prompt = prompt
        self.answer = answer

    @staticmethod
    def from_object(d):
        assert d["type"] == ShortAnswerQuestion.qtype
        prompt = d["prompt"]
        answer=  d["answer"]
        return ShortAnswerQuestion(prompt, answer)

    def get_answer_choices(self):
        return None


class TranscriptionAndParsingQuestion:
    qtype = "transcription-and-parsing"
    section_title = "Transcription and Parsing"

    def __init__(self, prompt, items, answers):
        self.prompt = prompt
        assert len(items) == len(answers)
        self.items = items
        self.answers = answers

    @staticmethod
    def from_object(d):
        assert d["type"] == TranscriptionAndParsingQuestion.qtype
        prompt = d["prompt"]
        items = d["items"]
        answers = d["answers"]
        return TranscriptionAndParsingQuestion(prompt, items, answers)

    def get_items_and_answers(self, shuffle=True):
        indices = [i for i in range(len(self.items))]
        if shuffle:
            random.shuffle(indices)
        return [self.items[i] for i in indices], [self.answers[i] for i in indices]

    def get_answer_choices(self):
        return None


class ExtraCreditWithImageQuestion:
    qtype = "extra-credit-with-image"
    section_title = "Extra Credit"

    def __init__(self, prompt, answer, image_fp):
        self.prompt = prompt
        self.answer = answer
        assert os.path.exists(image_fp), f"image file does not exist: {image_fp}"
        self.image_fp = image_fp

    @staticmethod
    def from_object(d):
        assert d["type"] == ExtraCreditWithImageQuestion.qtype
        prompt = d["prompt"]
        answer = d["answer"]
        image_fp = d["image"]
        return ExtraCreditWithImageQuestion(prompt, answer, image_fp)

    def get_answer_choices(self):
        return None


# if I keep using this code or make it a repo for others to use, then I should split out the extra-creditness of questions as a separate variable for modularity / ease of use
class ExtraCreditMultipleChoiceQuestion(MultipleChoiceQuestion):
    qtype = "extra-credit-multiple-choice"
    section_title = "Extra Credit"


class Question:
    types = {cls.qtype : cls for cls in [
        MultipleChoiceQuestion,
        TrueOrFalseQuestion,
        ShortAnswerQuestion,
        TranscriptionAndParsingQuestion,
        ExtraCreditWithImageQuestion,
        ExtraCreditMultipleChoiceQuestion,
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
        # "maroon-jackalope",
        # "forest-sasquatch",
        # "yellow-macadamia",
        # "indigo-dachshund",
        "加东叻沙",  # katong laksa
        "印尼炒饭",  # nasi goreng
        "加椰面包",  # kaya toast/bun
    ]
    version_names = random.sample(words, n_versions)
    versions = [exam.get_new_version(version_names[i]) for i in range(n_versions)]
    for i, v in enumerate(versions):
        exam_fp = os.path.join(out_dir, f"FinalExamVersion{i+1}.html")
        answer_fp = os.path.join(out_dir, f"FinalExamVersion{i+1}_Answers.html")
        with open(exam_fp, "w") as f:
            # print(f"\n---- version {i+1}, codename {version_names[i]} ----", file=f)
            v.print(file=f, with_answers=False)
        with open(answer_fp, "w") as f:
            v.print(file=f, with_answers=True)
