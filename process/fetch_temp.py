import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
  variables=["utide", "utotal", "vtide", "vtotal"],
  minimum_longitude=-45,
  maximum_longitude=37,
  minimum_latitude=25,
  maximum_latitude=72,
  start_datetime="2023-01-01T00:00:00",
  end_datetime="2025-01-01T00:00:00",
  # minimum_depth=0.49402499198913574,
  # maximum_depth=0.49402499198913574,
  output_filename="/home/stephen/PycharmProjects/Historical_Sea_Routing/process/data/copernicus/current_hourly_subset.zarr",
)