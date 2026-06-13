# Custom Map Layers Guide for Lithotheque

Lithotheque supports dynamic map overlays via WMS (Web Map Service) and XYZ tile protocols. Use this guide to quickly add geological maps from around the world.

## 🛠️ How to Add a Custom Layer

1. **Open the Map**: Navigate to the Map screen.
2. **Layer Control**: Tap the **Palette icon** (Layers).
3. **Add New**: Click the **"+"** button.
4. **Copy & Paste**: Use the blocks below to fill in the **Display Name**, **WMS URL**, and **Layer Name**.
5. **Set CRS**: Select the appropriate Coordinate Reference System (usually `EPSG:3857`).
6. **Activate**: Toggle the checkbox and adjust opacity.

---

## 📚 Recommended Map Sources (Alphabetical)

### A
#### ASG - African Surface Geology
Continental-scale geological overview of Africa.
- **Display Name:**
```text
ASG - African Surface Geology
```
- **WMS URL:**
```text
https://portal.onegeology.org/dynamic/wms
```
- **Layer Name:**
```text
Africa_Geology
```

#### Arizona Geological Survey
Official bedrock and surficial mapping for Arizona, USA.
- **Display Name:**
```text
Arizona Geological Survey
```
- **WMS URL:**
```text
http://services.azgs.az.gov/arcgis/services/aasg/AzGeology/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

---

### B
#### BGR - Germany Bedrock (250k)
Medium-scale geological coverage of Germany.
- **Display Name:**
```text
BGR - Germany Bedrock (250k)
```
- **WMS URL:**
```text
https://services.bgr.de/wms/geologie/gk250
```
- **Layer Name:**
```text
0
```

#### BGS - Bedrock Geology (UK)
Main rock types across the United Kingdom.
- **Display Name:**
```text
BGS - Bedrock Geology (UK)
```
- **WMS URL:**
```text
https://map.bgs.ac.uk/arcgis/services/BGS_Detailed_Geology/MapServer/WMSServer
```
- **Layer Name:**
```text
BGS_Detailed_Geology_Bedrock
```

#### BGS - Superficial Deposits (UK)
Surface materials and soft sediments across the UK.
- **Display Name:**
```text
BGS - Superficial Deposits (UK)
```
- **WMS URL:**
```text
https://map.bgs.ac.uk/arcgis/services/BGS_Detailed_Geology/MapServer/WMSServer
```
- **Layer Name:**
```text
BGS_Detailed_Geology_Superficial
```

#### BIG - Indonesia Geology
Unified geological mapping of the Indonesian archipelago.
- **Display Name:**
```text
BIG - Indonesia Geology
```
- **WMS URL:**
```text
https://geoportal.esdm.go.id/arcgis/services/Geology/Geological_Map_of_Indonesia/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### BRGM - France (1M)
National geological overview of France at 1:1,000,000.
- **Display Name:**
```text
BRGM - France (1M)
```
- **WMS URL:**
```text
https://geoservices.brgm.fr/geologie
```
- **Layer Name:**
```text
SCAN_F_GEOL1M
```

#### BRGM - France (50k)
Detailed geological mapping of France at 1:50,000.
- **Display Name:**
```text
BRGM - France (50k)
```
- **WMS URL:**
```text
https://geoservices.brgm.fr/geologie
```
- **Layer Name:**
```text
GEOLOGIE
```

#### BRGM - Hydrogeology (France)
Underground water and hydrogeological mapping for France.
- **Display Name:**
```text
BRGM - Hydrogeology (France)
```
- **WMS URL:**
```text
https://geoservices.brgm.fr/geologie
```
- **Layer Name:**
```text
SCAN_H_GEOL
```

#### British Columbia Geological Survey
Rock units and mineral data for BC, Canada.
- **Display Name:**
```text
British Columbia Geological Survey
```
- **WMS URL:**
```text
https://maps.gov.bc.ca/arcgis/services/mp_geology/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

---

### C
#### CGS - Czech Republic Bedrock
Official geological mapping for the Czech Republic.
- **Display Name:**
```text
CGS - Czech Republic Bedrock
```
- **WMS URL:**
```text
https://mapy.geology.cz/arcgis/services/Geofyzika/Magnetometrie/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### CGS - South Africa Geology
National 1:1,000,000 geological map of South Africa.
- **Display Name:**
```text
CGS - South Africa Geology
```
- **WMS URL:**
```text
http://geoscience.org.za/arcgis/services/Geology/Geology1M/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### CPRM - Brazil Geological Survey
Main geological mapping of Brazil at 1:250,000.
- **Display Name:**
```text
CPRM - Brazil Geological Survey
```
- **WMS URL:**
```text
https://geoservicos.cprm.gov.br/geoservicos/services/geologia/geologia_250k/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### CPRM - Brazil Minerals
Strategic mineral resources and deposits across Brazil.
- **Display Name:**
```text
CPRM - Brazil Minerals
```
- **WMS URL:**
```text
https://geoservicos.cprm.gov.br/geoservicos/services/recursos_minerais/recursos_minerais/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### California Geological Survey
Geological units and hazards for California, USA.
- **Display Name:**
```text
California Geological Survey
```
- **WMS URL:**
```text
https://maps.conservation.ca.gov/server/services/CGS/GeologicHazards/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

---

### G
#### GBA - Austria Geology
High-resolution geological coverage of Austria.
- **Display Name:**
```text
GBA - Austria Geology
```
- **WMS URL:**
```text
https://gis.geologie.ac.at/wms/geology
```
- **Layer Name:**
```text
GK500
```

#### GEBCO Bathymetric
Global ocean floor and underwater terrain mapping.
- **Display Name:**
```text
GEBCO Bathymetry
```
- **WMS URL:**
```text
https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/mapserv
```
- **Layer Name:**
```text
gebco_latest
```

#### GEUS - Denmark Geology
Surficial and bedrock data for Denmark.
- **Display Name:**
```text
GEUS - Denmark Geology
```
- **WMS URL:**
```text
https://data.geus.dk/arcgis/services/Geological_Map_of_Denmark_1_50000/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### GNS Science - New Zealand
Unified geological atlas of New Zealand.
- **Display Name:**
```text
GNS Science - New Zealand
```
- **WMS URL:**
```text
https://maps.gns.cri.nz/geology/wms
```
- **Layer Name:**
```text
NZL_GNS_2M_bedrock_geology
```

#### GSC Bedrock Geology (Canada)
The official national bedrock map of Canada.
- **Display Name:**
```text
GSC Bedrock Geology (Canada)
```
- **WMS URL:**
```text
https://gsc.nrcan.gc.ca/wms/geology_e
```
- **Layer Name:**
```text
Bedrock_Geology
```

#### GSC Surficial Geology (Canada)
Glacial and surface deposits mapping across Canada.
- **Display Name:**
```text
GSC Surficial Geology (Canada)
```
- **WMS URL:**
```text
https://gsc.nrcan.gc.ca/wms/surficial_e
```
- **Layer Name:**
```text
Surficial_Geology
```

#### GSI - India Geological Maps
National geological database for India.
- **Display Name:**
```text
GSI - India Geological Maps
```
- **WMS URL:**
```text
https://bhukosh.gsi.gov.in/arcgis/services/Geology/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### GSJ - Japan Geological Atlas
Standard national geological atlas of Japan.
- **Display Name:**
```text
GSJ - Japan Geological Atlas
```
- **WMS URL:**
```text
https://gbank.gsj.jp/ows/geology1m_en
```
- **Layer Name:**
```text
glg1m_en
```

#### GSJ - Japan Hydrogeology
Specialized groundwater mapping for Japan.
- **Display Name:**
```text
GSJ - Japan Hydrogeology
```
- **WMS URL:**
```text
https://gbank.gsj.jp/ows/hydrogeology
```
- **Layer Name:**
```text
hydrogeology
```

#### Geoscience Australia (Marine)
Ocean floor sediments mapping around Australia.
- **Display Name:**
```text
Geoscience Australia (Marine)
```
- **WMS URL:**
```text
https://services.ga.gov.au/gis/services/Marine_Sediment_Map_of_Australia/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### Geoscience Australia (Surface)
Unified surface geology of the Australian continent.
- **Display Name:**
```text
Geoscience Australia (Surface)
```
- **WMS URL:**
```text
https://services.ga.gov.au/gis/services/Surface_Geology_of_Australia_Optimised/MapServer/WMSServer
```
- **Layer Name:**
```text
Surface_Geology
```

#### Global Faults & Earthquakes (USGS)
Active faults and recent seismic activity visualization.
- **Display Name:**
```text
Global Faults & Earthquakes (USGS)
```
- **WMS URL:**
```text
https://earthquake.usgs.gov/arcgis/services/hazar/shakeraster/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

---

### I
#### IGME - Spain Geology (50k)
Detailed 1:50,000 scale geological mapping for Spain.
- **Display Name:**
```text
IGME - Spain Geology (50k)
```
- **WMS URL:**
```text
http://mapas.igme.es/gis/services/Cartografia_Geologica/IGME_Geologico_50/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### INGEMMET - Peru Geology
Regional rock units and geological structures for Peru.
- **Display Name:**
```text
INGEMMET - Peru Geology
```
- **WMS URL:**
```text
https://geocatminapp.ingemmet.gob.pe/arcgis/services/Geology/Geologia_Region/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### ISPRA - Italy Geology (100k)
Standard 1:100,000 scale geological coverage for Italy.
- **Display Name:**
```text
ISPRA - Italy Geology (100k)
```
- **WMS URL:**
```text
http://sgi.isprambiente.it/arcgis/services/servizi/geologia100k/MapServer/WMSServer
```
- **Layer Name:**
```text
Geologia
```

---

### M
#### Macrostrat World (XYZ)
High-performance global tiled geological map.
- **Display Name:**
```text
Macrostrat World (XYZ)
```
- **WMS URL:**
```text
https://tiles.macrostrat.org/carto/macrostrat/
```
- **Layer Name:**
```text
macrostrat
```

---

### N
#### NASA Earth Observations (Thermal)
Global land surface temperature and thermal anomalies.
- **Display Name:**
```text
NASA Earth Observations (Thermal)
```
- **WMS URL:**
```text
https://neo.gsfc.nasa.gov/wms/wms
```
- **Layer Name:**
```text
MOD_LSTD_M
```

#### NGU - Norway Bedrock
Official national bedrock mapping for Norway.
- **Display Name:**
```text
NGU - Norway Bedrock
```
- **WMS URL:**
```text
https://geo.ngu.no/mapserver/BedrockWMS
```
- **Layer Name:**
```text
Bedrock_Geology
```

---

### O
#### ONHYM - Morocco Minerals
Mineral deposits and geological resources for Morocco.
- **Display Name:**
```text
ONHYM - Morocco Minerals
```
- **WMS URL:**
```text
http://www.onhym.com/arcgis/services/Minerals/Mineral_Occurrences/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### OneGeology Global Bedrock (1M)
The unified global geological map at 1:1M scale.
- **Display Name:**
```text
OneGeology Global Bedrock (1M)
```
- **WMS URL:**
```text
http://portal.onegeology.org/dynamic/wms
```
- **Layer Name:**
```text
World_Geology
```

#### OneGeology Marine Geology
Global mapping of sea floor sediments and marine geology.
- **Display Name:**
```text
OneGeology Marine Geology
```
- **WMS URL:**
```text
http://portal.onegeology.org/dynamic/wms
```
- **Layer Name:**
```text
Sea_Floor_Sediments
```

#### Ontario Geological Survey
Central geological database for Ontario, Canada.
- **Display Name:**
```text
Ontario Geological Survey
```
- **WMS URL:**
```text
https://www.geologyontario.mndm.gov.on.ca/arcgis/services/OGS_Geology/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

---

### P
#### PGI - Poland Geological Map
Comprehensive geological mapping for Poland.
- **Display Name:**
```text
PGI - Poland Geological Map
```
- **WMS URL:**
```text
https://cbdg.pgi.gov.pl/arcgis/services/SMGP/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

---

### Q
#### Quebec Geological Survey (SIGÉOM)
Public geological database (SIGÉOM) for Quebec, Canada.
- **Display Name:**
```text
Quebec Geological Survey (SIGÉOM)
```
- **WMS URL:**
```text
https://sigeom.mines.gouv.qc.ca/arcgis/services/Sigeom_WMS/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

---

### S
#### SEGEMAR - Argentina Geology
National geological mapping of Argentina.
- **Display Name:**
```text
SEGEMAR - Argentina Geology
```
- **WMS URL:**
```text
http://servicios.segemar.gov.ar/arcgis/services/geologia/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### SERNAGEOMIN - Chile Geology
Bedrock and volcanic mapping for the Chilean territory.
- **Display Name:**
```text
SERNAGEOMIN - Chile Geology
```
- **WMS URL:**
```text
http://geoportal.sernageomin.cl/arcgis/services/Geologia/Geologia_Chile/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### SGC - Colombia Geology
The 2015 official unified geological map of Colombia.
- **Display Name:**
```text
SGC - Colombia Geology
```
- **WMS URL:**
```text
https://servicios.sgc.gov.co/arcgis/services/Geology/Mapa_Geologico_Colombia_2015/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

#### SGK - Switzerland Geology
Harmonized national geological atlas of Switzerland.
- **Display Name:**
```text
SGK - Switzerland Geology
```
- **WMS URL:**
```text
https://wms.geo.admin.ch/?
```
- **Layer Name:**
```text
ch.swisstopo.geologie-geologischer_atlas
```

#### SGU - Sweden Quaternary
Quaternary and surface deposits mapping for Sweden.
- **Display Name:**
```text
SGU - Sweden Quaternary
```
- **WMS URL:**
```text
https://resource.sgu.se/service/wms/130/jordarter-25k-100k
```
- **Layer Name:**
```text
Jordarter
```

---

### T
#### TNO - Netherlands Bedrock
Official subsurface geological mapping of the Netherlands.
- **Display Name:**
```text
TNO - Netherlands Bedrock
```
- **WMS URL:**
```text
https://services.bgs.ac.uk/tno/wms/geology
```
- **Layer Name:**
```text
Bedrock
```

#### Texas Bureau of Economic Geology
Rock units and geological resources for Texas, USA.
- **Display Name:**
```text
Texas Bureau of Economic Geology
```
- **WMS URL:**
```text
https://mapservice.beg.utexas.edu/server/services/Geology/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

---

### U
#### USGS - Coastal Geology (USA)
Specialized mapping of coastal and marine structures.
- **Display Name:**
```text
USGS - Coastal Geology (USA)
```
- **WMS URL:**
```text
https://cmgds.marine.usgs.gov/services/wms
```
- **Layer Name:**
```text
coastal_geology
```

#### USGS - Mineral Resources (USA)
Known mineral deposits and mining operations in the USA.
- **Display Name:**
```text
USGS - Mineral Resources (USA)
```
- **WMS URL:**
```text
https://mrdata.usgs.gov/services/mrds
```
- **Layer Name:**
```text
mrds
```

#### USGS - National Geologic Map (USA)
Unified geological map units for the entire United States.
- **Display Name:**
```text
USGS - National Geologic Map (USA)
```
- **WMS URL:**
```text
https://mrdata.usgs.gov/services/gmc
```
- **Layer Name:**
```text
USGS_GMC_Map_Units
```

#### USGS - Quaternary Faults (USA)
Database of active faults and seismic risk for the USA.
- **Display Name:**
```text
USGS - Quaternary Faults (USA)
```
- **WMS URL:**
```text
https://earthquake.usgs.gov/arcgis/services/hazar/qfaults/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

---

### W
#### Western Australia Geology
Highly detailed rock unit mapping for Western Australia.
- **Display Name:**
```text
Western Australia Geology
```
- **WMS URL:**
```text
https://services.slip.wa.gov.au/public/services/SLIP_Public_Services/Geology/MapServer/WMSServer
```
- **Layer Name:**
```text
0
```

---

*Note: WMS availability may vary based on provider maintenance. Always check the provider's official website for updated capabilities.*
