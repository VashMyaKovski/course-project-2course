from enum import Enum


class TextEncoding(str, Enum):
    UTF_8 = "utf-8"
    UTF_8_SIG = "utf-8-sig"
    CP1251 = "cp1251"
