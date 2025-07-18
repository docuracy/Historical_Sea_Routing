# eRutter: _Historical Sea Routing_

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

#### Geospatial Data Infrastructure

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

#### Historical Geographic Data

![Screenshot 3: Viabundus Water](/screenshots/viabundus_water.png)

The preprocessing pipeline includes Python scripts that support the application of spatial masks to reconcile
the contemporary geographic OpenStreetMap (OSM) coastlines with historical hydrographic reconstructions. For example,
the
[Viabundus](https://www.landesgeschichte.uni-goettingen.de/handelsstrassen/data/Viabundus-2-water-1500.geojson)
project's [Water (1500)](https://www.landesgeschichte.uni-goettingen.de/handelsstrassen/data/Viabundus-2-water-1500.geojson)
layer is applied in this way to mask land areas which have been reclaimed since the 16th century. A limitation is that
no oceanographic data are deployed in route-weighting for such areas.

#### Environmental Data

Environmental data plays a central role in the routing algorithm, representing the dynamic natural forces that
historically influenced maritime travel. Given the lack of detailed historical meteorological records, this project
employs carefully selected modern datasets as proxies for premodern conditions.

#### _Justification for Using Modern Environmental Data as a Proxy for Premodern Conditions_

Direct meteorological observations from premodern periods are sparse and geographically limited. To address this, the
model uses reanalysis and remote sensing data products to approximate past environmental conditions. Although climate
systems evolve, large-scale wind and wave patterns tend to exhibit significant stability at seasonal and regional scales
over decades and even centuries.

By focusing on modal (most typical) conditions, rather than averages or extreme events, this approach aligns well with
historical navigation needs:

* Routing decisions were shaped by dominant environmental conditions, not isolated anomalies.

* Modal climatologies highlight stable, recurring patterns that premodern mariners would have learned and exploited.

* The spatial scale of the H3 hexagonal grid aligns with modern dataset resolution and the generalised nature of
  historic vessel movement.

This proxy-based strategy supports practical, historically grounded simulation of maritime routes.

#### _Environmental Data Processing Overview_

This project uses variables from two Copernicus Marine Service datasets:

* [Global Ocean Hourly Reprocessed Sea Surface Wind and Stress from Scatterometer and Model](https://doi.org/10.48670/moi-00185) (
  0.125° resolution)
    * `eastward_wind`: Eastward component of wind vector
    * `northward_wind`: Northward component of wind vector
* [Global Ocean Physics Analysis and Forecast](https://doi.org/10.48670/moi-00016) (0.083° resolution, hourly)
    * `utotal`: Surface sea water x velocity
    * `vtotal`: Surface sea water y velocity

These datasets provide hourly wind and current measurements. Since current data are unavailable before June 2022, the
pipeline operates on a consistent two-year window (2023–2024).

* **Spatial Indexing:** Each H3 hexagonal graph node is mapped to the nearest environmental grid cell to ensure
  consistent spatial alignment.

* **Monthly Directional Flow Aggregation:** For each edge and calendar month, wind and current vector components are
  fetched for the edge midpoint, and projected onto the edge’s direction. The flow
  time series are classified into forward (positive projection) and reverse (negative projection) components. For each
  direction, the algorithm computes the weighted circular mean of angles and the arithmetic mean of magnitudes across
  all hourly observations. This approach captures the dominant directional flows separately for forward and reverse
  directions, reflecting realistic environmental conditions for routing.

#### _Visibility Reduction_

Modern meteorological data from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) are used to
estimate historical attenuation of visibility due to fog and rain.

### Dynamic Processing

The core routing logic is based on the Dijkstra bidirectional shortest path algorithm
from [graphology](https://graphology.github.io/),
with weights calculated dynamically via the `premodern-sailing.js` module.

This module is a prototype implementation that estimates traversal cost (time) over each edge by factoring
in:

- **Distance**: Great-circle distance between source and target hex centroids.
- **Wind direction and speed**: Modal characteristics per month.
- **Current direction and speed**: Current data (the phase closest to the edge direction is used).
- **Vessel parameters**: Sourced from `premodern-sailing.js`, including draught, beam, and sail characteristics, and
  dynamically adjusted depending on the loaded weight.
- **Bathymetry constraints**: Nodes in shallow waters incur heavy penalties if the depth falls below vessel
  draught tolerance.

A custom weight function uses these inputs to simulate the effective time taken by a given sailing vessel across an edge
for a given month. The route finder adapts to seasonal conditions, allowing month-specific simulations of outward and
return legs.

### Longevity

A primary goal was to enable users to explore historical maritime routes entirely within the browser, without relying on
server-side processing or infrastructure. This approach protects the project from disruptions caused by changes in
funding, hosting, or technical support, problems which are common in DH projects.

## Coverage

Coverage is currently limited to Europe (as shown in the map at [Preprocessing](#preprocessing)), but can be extended to
other areas by running the included Python
scripts. A lighter-weight subgraph covering only the UK and Ireland can be loaded by appending `?aoi=UK-Eire` to the
URL.

![Screenshot 4: UK+Eire Subgraph](/screenshots/uk_eire.png)

## Limitations

- The in-browser processing approach can be resource-intensive for large datasets, potentially causing performance
  issues on less powerful devices.
- Loading very large graphs may result in significant delays or browser memory exhaustion.
- The lack of server-side support limits the ability to perform real-time data updates.
- Currently supports only certain maritime regions.
- The routing and estimation algorithms rely on simplified models that may not capture all historical navigational
  nuances.
- User interface and visualisation features may require further refinement for accessibility and ease of use.

## Calibration

The results generated by this tool should be checked against historical records of journey times and itineraries
where available, and the parameters of the vessel models should be adjusted accordingly.

> 💡 Contributions welcome: If you have access to historical shipping logs, port books, or travel diaries that document
> journey durations or routes, please consider submitting them (or links to them) via GitHub Issues or Pull Requests.
> Data from merchant, naval, or fishing vessels dated before 1700 and covering the North Sea, Baltic, English Channel,
> Eastern Atlantic, and Mediterranean regions would be particularly valuable.

For example:

- [Journal of Alexander Gillespie, skipper in Elie (1662-1685)](https://collections.st-andrews.ac.uk/item/journal-of-alexander-gillespie-skipper-in-elie/2078154)

  > "This journal records voyages undertaken by Alexander Gillespie. It contains general information about cargoes,
  ports and lengths of journeys; ... Gillespie's main voyages were one into the Baltic in the early summer, one to
  Bordeaux in the autumn for the first vintage, and occasionally a second."

- [Henry Teonge’s Diary, 1675–1676](https://babel.hathitrust.org/cgi/pt?id=hvd.hxjrrd&seq=11)

  > Teonge, an English naval chaplain, recorded daily positions, weather, and course during his voyages to the
  Mediterranean and Levant between June 1675 and November 1676, including leg-by-leg progress estimates.

See also:

- Pryor, J.H. (1988) ‘The ships’, in _Geography, Technology, and War: Studies in the Maritime History of the
  Mediterranean, 649–1571_. Cambridge: Cambridge University Press (Past and Present Publications), pp. 25–86.
  DOI: https://doi.org/10.1017/CBO9780511562501.006.

  > “Travelling in a cog in 1384, three Tuscan pilgrims took 23 days for the voyage from Venice to Alexandria but 42 for
  the return to Venice from Beirut. On board a Venetian great galley in 1395, Ogier VIII d’Anglure took 32 days to reach
  Jaffa but over five months for the return to Venice. The galley on which Felix Fabri travelled in 1480 took 43 days to
  make Jaffa from Venice but 70 for the return voyage.”

  > “Two spring voyages from Genoa to Alexandria in 1379 and 1391 took 24 days and 35 days respectively, whereas a late
  autumn voyage from Beirut to Genoa in 1396 took 53 days.”

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