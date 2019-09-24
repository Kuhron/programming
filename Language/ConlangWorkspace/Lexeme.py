from InflectionForm import InflectionForm
from Word import Word


class Lexeme:
    def __init__(self, citation_form, part_of_speech, gloss, forms=None, inflection_forms=None):
        self.citation_form = citation_form
        assert part_of_speech.isidentifier(), "part of speech \"{}\" is not a valid identifier".format(part_of_speech)
        self.part_of_speech = part_of_speech
        self.parse_gloss(gloss)

        if forms is None:
            self.create_forms(inflection_forms)
        else:
            self.forms = forms
        self.validate_forms()

    def validate_forms(self):
        for form in self.forms:
            assert type(form) is Word and form.has_designation(), "invalid form: {}".format(form)

    def create_forms(self, inflection_forms):
        self.forms = []
        for inf in inflection_forms:
            form = self.apply_inflection_form(inf)
            assert form.has_designation(), "form has no designation: {}".format(form)
            self.forms.append(form)

    def apply_inflection_form(self, inflection):
        citation_form = self.citation_form
        assert type(citation_form) is Word and type(inflection) is InflectionForm
        new_string = inflection.string.replace("-", citation_form.to_str())
        new_designation = citation_form.designation + inflection.designation_suffix
        new_gloss = self.short_gloss + inflection.gloss
        new_word = Word.from_str(new_string, designation=new_designation)
        new_word.gloss = new_gloss
        return new_word

    def parse_gloss(self, gloss_str):
        if gloss_str[:2] == "g:":
            # has short gloss
            short_gloss, *rest = gloss_str.split(" ")
            self.short_gloss = short_gloss[2:]
            self.gloss = " ".join(rest)
        else:
            self.short_gloss = self.gloss = gloss_str
