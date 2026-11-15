"""tiles.region_of on well-known places (coarse boxes; documented as approximate)."""
import pytest
from nightrad.data.tiles import region_of

CASES = [("Riau Indonesia", 0.5, 101.5, "AS"), ("Amazon Para", -4, -52, "SA"), ("Cerrado", -13, -47, "SA"), ("California", 38, -121, "NA"),
         ("Alaska", 64, -150, "NA"), ("Siberia", 60, 110, "AS"), ("Kazakhstan", 48, 68, "AS"), ("Portugal", 40, -8, "EU"), ("Ukraine", 49, 32, "EU"),
         ("Angola", -12, 17, "AF"), ("Zambia", -14, 28, "AF"), ("Sahel Mali", 14, -2, "AF"), ("South Africa", -30, 25, "AF"), ("Madagascar", -19, 47, "AF"),
         ("N Australia", -14, 131, "OC"), ("Queensland", -24, 148, "OC"), ("New Zealand", -42, 172, "OC"), ("Papua New Guinea", -6, 145, "OC"),
         ("India", 22, 78, "AS"), ("Myanmar", 20, 96, "AS"), ("Mexico", 20, -102, "NA"), ("Colombia", 5, -74, "SA"), ("Korea", 36.5, 127.5, "AS")]


@pytest.mark.parametrize("name,lat,lon,want", CASES)
def test_region_of(name, lat, lon, want):
    assert region_of(lat, lon) == want, name
