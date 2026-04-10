from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType
from datetime import datetime
import numpy as np

def get_real_ndvi(polygon_coords, client_id, client_secret):
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret

    # Extract lat/lon from polygon
    lons = [p[0] for p in polygon_coords]
    lats = [p[1] for p in polygon_coords]

    min_lon = min(lons)
    max_lon = max(lons)
    min_lat = min(lats)
    max_lat = max(lats)

    bbox = BBox(
        bbox=[min_lon, min_lat, max_lon, max_lat],
        crs=CRS.WGS84
    )

    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B08"],
            output: {
                bands: 1,
                sampleType: "FLOAT32"
            }
        }
    }

    function evaluatePixel(sample) {
        let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
        return [ndvi];
    }
    """

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=("2024-01-01", datetime.today().strftime("%Y-%m-%d"))
            )
        ],
        responses=[
            SentinelHubRequest.output_response("default", MimeType.TIFF)
        ],
        bbox=bbox,
        size=(100, 100),
        config=config
    )

    data = request.get_data()[0]

    ndvi = float(np.nanmean(data))

    # Safety clamp
    ndvi = max(-1, min(1, ndvi))

    return ndvi