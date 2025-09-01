"""Weather information tools."""
import requests
from langchain.tools import tool


@tool
def get_weather(city: str, language: str = "pt") -> str:
    """
    Obtém informações meteorológicas atuais para uma cidade usando Open-Meteo API.
    
    Args:
        city: Nome da cidade para consulta
        language: Idioma para a consulta (padrão: pt)
    
    Returns:
        Informações do clima atual da cidade
    """
    try:
        # 1) Geocoding
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "count": 1,
            "language": language.lower(),
            "format": "json",
        }
        
        geo_response = requests.get(geocode_url, params=geo_params, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Não encontrei coordenadas para '{city}'."

        place = geo_data["results"][0]
        lat, lon = place["latitude"], place["longitude"]

        # 2) Weather data
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        forecast_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        }

        forecast_response = requests.get(forecast_url, params=forecast_params, timeout=10)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        current = forecast_data.get("current", {})
        
        return f"""Clima em {place.get("name")}, {place.get("country")}:
- Temperatura: {current.get("temperature_2m")}°C
- Umidade: {current.get("relative_humidity_2m")}%
- Vento: {current.get("wind_speed_10m")} km/h
- Coordenadas: {lat}, {lon}
Fonte: open-meteo.com"""

    except Exception as e:
        return f"Erro ao buscar informações do clima: {str(e)}"


tools = [get_weather]