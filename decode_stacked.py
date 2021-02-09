import re
from json import JSONDecoder, JSONDecodeError


def decode_stacked(document, pos=0, decoder=JSONDecoder()):
    try:
        NOT_WHITESPACE = re.compile(r'[^\s]')
        while True:
            match = NOT_WHITESPACE.search(document, pos)
            if not match:
                return
            pos = match.start()

            try:
                obj, pos = decoder.raw_decode(document, pos)
            except JSONDecodeError:
                print("decode_stacked problem")
            yield obj
    except:
        print("decode_stacked failed")