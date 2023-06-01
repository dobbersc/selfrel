import math
import sqlite3


def function_is_registered(cursor: sqlite3.Cursor, function_name: str, number_of_arguments: int) -> bool:
    """
    Checks if a function is registered in the database.
    :param cursor: A database cursor
    :param function_name: The name of the function to be checked
    :param number_of_arguments: The number of arguments of the function to be checked
    :return: True, if the specified function exists in the database. Otherwise, False.
    """
    exists: int = cursor.execute(
        "SELECT exists(SELECT 1 FROM pragma_function_list WHERE name = ? AND narg = ?)",
        (function_name, number_of_arguments),
    ).fetchone()[0]
    return bool(exists)


def register_log(connection: sqlite3.Connection) -> None:
    """
    Registers the `log` function according to the built-in mathematical function of sqlite3.
    If the function has been registered already, this function has no side effects.
    Reference: https://www.sqlite.org/lang_mathfunc.html#log
    """
    cursor: sqlite3.Cursor = connection.cursor()
    if not function_is_registered(cursor, function_name="log", number_of_arguments=1):
        connection.create_function("log", narg=1, func=math.log10, deterministic=True)
    if not function_is_registered(cursor, function_name="log", number_of_arguments=2):
        connection.create_function("log", narg=2, func=lambda b, x: math.log(x, b), deterministic=True)
