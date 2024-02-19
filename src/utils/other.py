def change_name_with_type(name: str, new_part_to_append: str) -> str:
    """
    preserve pattern with letter at the end

    Args:
        name (str): col name
        new_part_to_append (str):

    Returns:
        str: new name
    """
    return name[:-1] + new_part_to_append + name[-1]