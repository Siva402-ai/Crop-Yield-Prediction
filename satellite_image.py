from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType
from datetime import datetime
import numpy as np
from PIL import Image


def download_satellite_image(polygon_coords, client_id, client_secret):

    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret

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
            input: ["B04","B03","B02"],
            output: { bands: 3 }
        }
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
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
            SentinelHubRequest.output_response("default", MimeType.PNG)
        ],
        bbox=bbox,
        size=(1024,1024),
        config=config
    )

    img = request.get_data()[0]

    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)

    image = Image.fromarray(img)

    image.save("farm_satellite.png")

    return "farm_satellite.png"