import unittest

from nwp.core.config import load_bundle, to_plain


class DataContractTests(unittest.TestCase):
    def test_archive_forecast_contract_keeps_total_cloud_only(self):
        variables = to_plain(load_bundle())["data"]["forecast"]["variables"]
        self.assertEqual(len(variables), 15)
        self.assertIn("cloud_cover", variables)
        for field in ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"):
            self.assertNotIn(field, variables)


if __name__ == "__main__":
    unittest.main()
