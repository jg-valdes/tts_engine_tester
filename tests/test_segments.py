import unittest

from src.runner import load_segments_payload, parse_time_value


class SegmentParsingTests(unittest.TestCase):
    def test_parse_integer_milliseconds(self):
        self.assertEqual(parse_time_value(1500), 1500)
        self.assertEqual(parse_time_value("1500"), 1500)

    def test_parse_hh_mm_ss_millis(self):
        self.assertEqual(parse_time_value("00:00:05.250"), 5250)
        self.assertEqual(parse_time_value("00:00:05,25"), 5250)

    def test_parse_hh_mm_ss_centiseconds(self):
        self.assertEqual(parse_time_value("00:00:05:00"), 5000)
        self.assertEqual(parse_time_value("00:00:13:50"), 13500)

    def test_load_segments_normalizes_string_times_to_ms(self):
        segments = load_segments_payload(
            {
                "segments": [
                    {
                        "id": "s1",
                        "startTime": "00:00:05:00",
                        "endTime": "00:00:13:50",
                        "description": "Example line",
                    }
                ]
            }
        )

        self.assertEqual(segments[0].start_time, 5000)
        self.assertEqual(segments[0].end_time, 13500)

    def test_invalid_time_format_raises(self):
        with self.assertRaises(ValueError):
            parse_time_value("5 seconds")


if __name__ == "__main__":
    unittest.main()
