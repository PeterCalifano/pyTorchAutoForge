from __future__ import annotations

import threading

import numpy as np

from pyTorchAutoForge.api.tcp import (
    DataProcessor,
    ProcessingMode,
    pytcp_requestHandler,
    pytcp_server,
)


def _Identity(input_data_: np.ndarray) -> np.ndarray:
    return input_data_


def test_data_processor_tensor_roundtrip() -> None:
    processor_ = DataProcessor(
        _Identity,
        np.float32,
        1024,
        ENDIANNESS="little",
        DYNAMIC_BUFFER_MODE=True,
        PRE_PROCESSING_MODE=ProcessingMode.TENSOR,
    )
    input_array_ = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    output_buffer_ = processor_.process(
        processor_.TensorToBytesBuffer(input_array_))
    output_array_, output_shape_ = processor_.BytesBufferToTensor(output_buffer_[
                                                                  4:])

    assert output_shape_ == input_array_.shape
    assert np.array_equal(output_array_, input_array_)


def test_tcp_server_starts_and_shuts_down_without_hanging() -> None:
    processor_ = DataProcessor(
        _Identity,
        np.float32,
        1024,
        ENDIANNESS="little",
        DYNAMIC_BUFFER_MODE=True,
        PRE_PROCESSING_MODE=ProcessingMode.TENSOR,
    )

    with pytcp_server(
        ("127.0.0.1", 0),
        pytcp_requestHandler,
        processor_,
        bindAndActivate=True,
    ) as server_:
        thread_ = threading.Thread(target=server_.serve_forever, daemon=True)
        thread_.start()

        assert server_.server_address[0] == "127.0.0.1"
        assert server_.server_address[1] > 0

        server_.shutdown()
        thread_.join(timeout=2.0)

    assert not thread_.is_alive()
