# Historical Sailing Routes

This repository provides an interactive web-based tool for exploring historical sailing routes using environmental data
and graph-based routing algorithms. It combines spatial and temporal environmental datasets to estimate plausible
maritime routes based on seasonal and environmental conditions. The tool may be useful for historians, geographers, and
researchers interested in maritime history, environmental impacts on sailing, and historical route reconstructions.


> ⚠️ **Note:** Both the vessel characteristics (`sailing_vessels.js`) and the sailing-time algorithm (`sailing.js`) are
> proof-of-concept and
> require refinement. Contributions are welcome, especially from domain experts with knowledge of historical or
> contemporary
> sailing vessel performance and environmental interactions.

![Screenshot 1: Route visualisation on map](/screenshots/routes_and_parameters.png)

- Vessel parameters can be loaded from
  a selection of preset vessel types, and adjusted manually.
- Return voyages can also be included.
- Auto-cycling of months facilitates exploration of seasonal route
  variations.
- Voyage distances and durations are logged for both outward and
  return journeys.
- The routes, logs, parameters, and data sources can be exported as GeoJSON for further analysis and visualisation.

## Demo

The Route Explorer is online here:
https://docuracy.github.io/Historical_Sea_Routing

## Methodology

Data processing in this project is divided into two principal stages: **preprocessing** and **dynamic (browser-based)
processing**.

The **preprocessing** stage is performed using a suite of Python scripts that acquire, transform, and integrate multiple
geospatial datasets. These include satellite-derived environmental variables, elevation models, and historical
geographic overlays. This stage is responsible for constructing the hexagonal grid infrastructure, associating
environmental attributes with each node, and computing theoretical visibility ranges. By performing these calculations
in advance, the system ensures that complex spatial relationships are efficiently encoded and ready for real-time
exploration.

The **dynamic processing** stage occurs entirely within the browser and is powered by JavaScript. It enables users to
interactively compute and visualise plausible sailing routes in near real time, based on selected vessel profiles and
seasonal environmental conditions. The browser-side logic includes an experimental cost-weighting algorithm that
estimates travel time between nodes by incorporating wind, currents, draught constraints, and weather-induced visibility
limitations.

This bifurcated processing architecture provides a scalable and extensible framework: the computationally intensive
tasks are handled offline during preprocessing, while lightweight, user-driven analyses are performed on demand in the
browser.

### Preprocessing

![Screenshot 2: Multi-Resolution Hex Grid](/screenshots/hex_grid.png)  
This project leverages the [H3 hexagonal hierarchical spatial index](https://h3geo.org/) to create a multi-resolution
grid system for
representing geographic areas. H3 provides consistent spatial coverage with hexagons at multiple resolutions, enabling
scalable
routing and environmental analysis. The blending of multiple resolutions allows minimisation of the graph size.

![Screenshot 3: Land-Sight Computation](/screenshots/land_sight.png)  
For sight-line computations, H3 cells at resolution 5 are placed over
land, and Digital Elevation Model (DEM) data are clipped to these hex boundaries.
The maximum elevation within each land cell is retained. To model visibility from sea, the horizon distance for each sea
cell is computed using Earth curvature geometry and the
maximum
land elevations. A radial comparison identifies which land cells are theoretically visible from each sea node, stopping
at the
first visible landmass within the horizon radius. The efficiencies of this approach allow for rapid sight-line analysis
without recourse to ray-casting.

Modern meteorological data from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) are used to
estimate historical attenuation of visibility due to fog and rain.

The preprocessing pipeline includes Python scripts that support the application of spatial masks to reconcile
the contemporary geographic OpenStreetMap (OSM) coastlines with historical hydrographic reconstructions. For example,
the
[Viabundus](https://www.landesgeschichte.uni-goettingen.de/handelsstrassen/data/Viabundus-2-water-1500.geojson)
project's [Water (1500)](https://www.landesgeschichte.uni-goettingen.de/handelsstrassen/data/Viabundus-2-water-1500.geojson)
layer is applied in this way to mask land areas which have been reclaimed since the 16th century.

### Dynamic Processing

The core routing logic is based on the Dijkstra bidirectional shortest path algorithm
from [graphology](https://graphology.github.io/),
with weights calculated dynamically via `sailing.js`.

The `sailing.js` module is a prototype implementation that estimates traversal cost (time) over each edge by factoring
in:

- **Distance**: Great-circle distance between source and target hex centroids.
- **Wind direction and speed**: Monthly averages, influencing effective sailing angle and speed.
- **Sea surface conditions**: Wave height and surface currents, where applicable.
- **Vessel parameters**: Sourced from `sailing_vessels.js`, including draught, beam, and ideal points of sail.
- **Bathymetry constraints**: Nodes or edges in shallow waters incur heavy penalties if the depth falls below vessel
  draught tolerance.

A custom weight function uses these inputs to simulate the effective time taken by a given sailing vessel across an edge
for a
specific month. The route finder adapts to seasonal conditions, allowing month-specific simulations of outward and
return legs.

## Coverage

Coverage is currently limited to Europe (as shown in the map at [Preprocessing](#preprocessing)), but can be extended to other areas by running the included Python
scripts. A lighter-weight subgraph covering the UK and Ireland can be loaded by appending `?aoi=UK-Eire` to the URL.

![Screenshot 3: UK+Eire Subgraph](/screenshots/uk_eire.png)

## References and Data Sources

This project builds upon and is informed by the following key references:

- Holterman, Bart. "14 Sources and methods for the reconstruction of medieval and early modern sea routes in northern
  Europe". _Mobility in the Early Middle Ages, and Beyond – Mobilität im Frühmittelalter und darüber hinaus:
  Interdisciplinary Approaches – Interdisziplinäre Zugänge_, edited by Laury Sarti and Helene von Trott zu Solz, Berlin,
  Boston: De Gruyter, 2025, pp. 287-306. https://doi.org/10.1515/9783111166698-014
- Litvine, A.D., Lewis, J. & Starzec, A.W. A multi-criteria simulation of European coastal shipping routes in the ‘age
  of sail’. _Humanit Soc Sci Commun_ **11**, 666 (2024). https://doi.org/10.1057/s41599-024-02906-9

### Environmental Data Sources

- Copernicus Marine Environment Monitoring Service (CMEMS) datasets, including bathymetry, wave, and wind data.
- ERA5 reanalysis weather datasets from the Copernicus Climate Data Store.
- Digital Elevation Model (DEM) data from Mapzen's Terrarium tiles.

---
© 2025 Stephen Gadd

This work is licensed under the [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).