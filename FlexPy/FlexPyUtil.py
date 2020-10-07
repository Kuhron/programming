def get_single_child(element, child_tag):
    children = element.findall(child_tag)
    if len(children) == 1:
        return children[0]
    error_str = "not one {} was found, but {}:\n".format(child_tag, len(children))
    for child in children:
        error_str += "<{}> tag with attributes {}\n".format(child.tag, child.attrib)
    error_str += "in parent:\n<{}> tag with attributes {}".format(element.tag, element.attrib)
    raise Exception(error_str)



