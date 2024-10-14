import pytest


def inc(x):
    return x + 1
def test_answer():
    assert inc(3) == 5


def main():
    # Define example unit test
    def inc(x):
        return x + 1
    def test_answer():
        assert inc(3) == 5
    # Run the test (call from shell)
    pytest.main(['-x', __file__])

if __name__ == '__main__':
    main()