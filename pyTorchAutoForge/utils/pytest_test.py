import pytest

# Example using simple assert --> pytest shows traceback
def inc(x):
    return x + 1
def test_1():
    assert inc(4) == 5

def test_2():
    assert inc(4) == 5

# Example using pytest.raises for specific exceptions
def f():
    raise SystemExit(1)

def test_3():
    with pytest.raises(SystemExit):
        f()

def f_2():
    raise ExceptionGroup(
        "Group message",
        [
            RuntimeError(),
        ],
    )

def test_exception_in_group():
    with pytest.raises(ExceptionGroup) as excinfo:
        f_2()
    assert excinfo.group_contains(RuntimeError)
    assert not excinfo.group_contains(TypeError)


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