"""
weather.py - Weather tool using Open-Meteo API (free, no key required).
Geocodes city name to coordinates, then fetches current weather.
Handles diacritics and comma-separated city names.
"""

import unicodedata
import requests
from langchain_core.tools import tool


def _normalize(text: str) -> str:
    """Strip diacritics and take only the city name (before comma)."""
    # Take first part if comma-separated (e.g., "Willemstad, Curaçao" → "Willemstad")
    city_name = text.split(",")[0].strip()
    # Remove diacritics (e.g., ç → c)
    nfkd = unicodedata.normalize("NFKD", city_name)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    try:
        # Try original city name first, then normalized version
        attempts = [city, _normalize(city)]
        geo_result = None

        for attempt in attempts:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={attempt}&count=1"
            geo = requests.get(geo_url).json()
            if geo.get("results"):
                geo_result = geo["results"][0]
                break

        if not geo_result:
            return f"Could not find location: {city}"

        lat = geo_result["latitude"]
        lon = geo_result["longitude"]

        # Fetch current weather
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        weather = requests.get(weather_url).json()
        cw = weather["current_weather"]

        return (
            f"Weather in {city}: {cw['temperature']}°C, "
            f"wind {cw['windspeed']} km/h, "
            f"weather code {cw['weathercode']}"
        )
    except Exception as e:
        return f"Error fetching weather for {city}: {e}"
