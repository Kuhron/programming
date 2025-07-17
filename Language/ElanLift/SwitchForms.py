# for Erin SanGregory dealing with Wakhi lexicon in ELAN
# note that you have to import this .lift as lexicon in ELAN and then go to the lexicon properties, custom fields, add the "Orthographic" field under entry yourself, and then it will show up. Somehow ELAN doesn't recognize that it needs to show this field until you manually create it in the GUI.


from xml.etree import ElementTree as ET
import sys


fp = "WblLift_original.lift"
with open(fp) as f:
    lines = f.readlines()

tree = ET.parse(fp)
root = tree.getroot()

dummy_ipa_form = ET.Element("form", attrib={"lang": "wbl-fonipa"})
dummy_ipa_text = ET.Element("text")
dummy_ipa_text.text = ""
dummy_ipa_form.append(dummy_ipa_text)

dummy_arabic_form = ET.Element("form", attrib={"lang": "wbl-Arab"})
dummy_arabic_text = ET.Element("text")
dummy_arabic_text.text = ""
dummy_arabic_form.append(dummy_arabic_text)


def mutate_to_put_ipa_first(el):
    children = [x for x in el]
    forms = el.findall("form")

    assert len(forms) < 3, "too many forms!"
    assert len(forms) > 0, "no forms!"

    if not any(form.attrib["lang"] == "wbl-fonipa" for form in forms):
        has_ipa = False
        print(f"no IPA in entry with guid {entry.attrib['guid']}")
        ipa_form = dummy_ipa_form
    else:
        has_ipa = True
        ipa_form ,= [form for form in forms if form.attrib["lang"] == "wbl-fonipa"]

    if not any(form.attrib["lang"] == "wbl-Arab" for form in forms):
        has_arabic = False
        print(f"no Arabic in entry with guid {entry.attrib['guid']}")
        arabic_form = dummy_arabic_form
    else:
        has_arabic = True
        arabic_form ,= [form for form in forms if form.attrib["lang"] == "wbl-Arab"]

    # put the form children of lexical-unit in the correct order
    for x in forms:  # don't do x in el, nogut change size during iteration
        assert x.tag == "form", f"extra child of <lexical-unit> found with tag <{x.tag}>"
        el.remove(x)
    el.append(ipa_form)
    el.append(arabic_form)
    return el, ipa_form, arabic_form


header ,= root.findall("header")
fields_el ,= header.findall("fields")
orthographic_field_el = ET.Element("field", attrib={"tag": "Orthographic"})
ortho_form_el = ET.Element("form", attrib={"lang": "en"})
ortho_text_el = ET.Element("text")
ortho_text_el.text = "This field stores the Arabic-script orthography for the entry."
ortho_form_el.append(ortho_text_el)
orthographic_field_el.append(ortho_form_el)
fields_el.append(orthographic_field_el)

for child in root:
    if child.tag != "entry":
        continue
    entry = child
    lexical_unit ,= entry.findall("lexical-unit")
    lexical_unit, ipa_form, arabic_form = mutate_to_put_ipa_first(lexical_unit)
    variants = entry.findall("variant")
    for variant in variants:
        variant, ipa_form, arabic_form = mutate_to_put_ipa_first(variant)

    # want the orthographic field to be a child of entry, not of lexical-unit
    orthographic = ET.Element("field", attrib={"type": "Orthographic"})
    orthographic.append(arabic_form)
    entry.append(orthographic)


output_fp = "WblLift_output.lift"
tree.write(output_fp, encoding="utf-8", short_empty_elements=True, xml_declaration=True)

