

# AquaEV: Flood-Aware EV Journey Planning for South Wales

## 🌍 Research Highlights
AquaEV is a decision-support dashboard designed to help EV users in South Wales plan safe journeys during flood events.  
The dashboard integrates:
- Welsh Government flood-risk maps (FRAW, FMfP, live warnings)  
- UK National ChargePoint Registry charging station data  
- Weather forecasts (Met Office DataHub, Open-Meteo, 24-hour)  

Users can input journey destinations, battery charge status, and generate downloadable routes.

---

## 🎯 Science Objective
To integrate and visualise diverse datasets for South Wales (UK), enabling a decision-support dashboard that:  
- Supports **stakeholder planning** for resilient infrastructure.  
- Allows **EV users** to self-serve real-time data for safe journey planning during extreme events.  

---

## ⚙️ Approach
1. **Data Sources**
   - EV chargers: UK National ChargePoint Registry (filtered to South Wales).  
   - Flood data: Welsh Government GeoServer (FRAW, FMfP, live warnings from NRW).  
   - Weather data: Met Office DataHub and Open-Meteo.  

2. **Integration**
   - Semantic Knowledge Graphs fuse heterogeneous datasets (EV, flood, weather).  
   - Conversational AI enables natural-language queries, scenario exploration, and transparent explanations.  

3. **Routing Engine**
   - **Exact mode:** Resource-Constrained Shortest Path (RCSP) solver over OSMnx-derived graphs.  
   - **Fallback mode:** Open Source Routing Machine (OSRM) API.  
   - Penalties applied to flood-exposed road segments.  

---

## 🚗 Scenario Example
**Inputs:**  
- Start: Cardiff City Centre  
- Destination: Swansea Marina  
- Battery: 40% (≈90 km range)  

**Steps:**  
- Direct motorway route (M4) flagged due to flood closure near Port Talbot.  
- Simulator queries charging stations on alternative routes.  
- Suggests diversion via A48 with safe recharge stop at Bridgend (outside flood zone).  

**Chatbot explanation:**  
> “Direct route via M4 has flood closure near Port Talbot. Based on your battery status (40%), I recommend diverting via A48 and recharging at Bridgend station, which is currently outside flood risk zones.”  

---

## 📊 Impact
- Advances **flood-aware transport planning** by integrating EV, flood, and weather data.  
- Optimises routing and charging strategies under extreme weather.  
- Provides **transparent, natural-language insights** via chatbot for planners and drivers.  
- Empowers drivers with **real-time decision support** to balance energy and risk.  
- Informs **data-driven investments** and emergency planning.  
- Offers a **transferable framework** for other flood-prone regions worldwide.  

---

## 📝 Summary
AquaEV integrates EV charging, flood, and weather data into a unified dashboard.  
It improves **resilience in South Wales**, supports **real-time driver safety**, and provides **evidence-based insights** for planners.  
The approach is scalable and can be adapted to other regions.  

---

## 📂 Repository Structure



<p align="center">
  <img src="aquaev-dashboard.png" alt="AquaEV — Decision Support Dashboard" width="900">
  <br><em>Figure: AquaEV dashboard with flood overlays, journey simulator, and 24-hour forecast.</em>
</p>
