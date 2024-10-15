import pytest


def inc(x):
    return x + 1
def test_1():
    assert inc(4) == 5

def test_2():
    assert inc(3) == 5

# DEVNOTE: pytest does NOT run test scripts as main. The test asserts must be reachable outside other functions.
# All functions with test* in their name are executed as unit test.
def main():
    # Define example unit test
    def inc(x):
        return x + 1
    def test_answer():
        assert inc(3) == 5

    test_answer(4) # This never runs
    # Run the test (call from shell)
    pytest.main(['-x', __file__])

if __name__ == '__main__':
    main()