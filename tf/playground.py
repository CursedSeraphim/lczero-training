from shufflebuffer import ShuffleBuffer

def test_extract(self):
        sb = ShuffleBuffer(3, 1)
        r = sb.extract()
        assert r == None, r  # empty buffer => None
        r = sb.insert_or_replace(b'111')
        assert r == None, r  # buffer not yet full => None
        r = sb.extract()
        assert r == b'111', r  # one item in buffer => item
        r = sb.extract()
        assert r == None, r  # buffer empty => None

sb = ShuffleBuffer(1,1)
r = sb.insert_or_replace(b'1')
print(r)

r = sb.insert_or_replace(b'2')
print(r)

r = sb.insert_or_replace(b'3')
print(r)

r = sb.extract()
print(r)
