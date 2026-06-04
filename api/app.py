import joblib
import pandas as pd
import numpy as np
import re
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')


def build_dataframe(data):
    """
    Replicates all feature engineering from the notebook before prediction.
    """
    # ── core fields ──────────────────────────────────────────────
    bedrooms = max(float(data.get('bedrooms', 1) or 1), 1)
    guests   = float(data.get('guests', 2) or 2)
    beds     = float(data.get('beds', bedrooms) or bedrooms)
    baths    = float(data.get('baths', 1) or 1)

    # ── amenities: accept either a count (int) or raw comma-separated string ──
    amenities_raw = str(data.get('amenities_raw', '') or '')
    amenities_count = int(data.get('amenities_count', 0) or 0)
    if amenities_raw:
        amenities_count = len([a for a in amenities_raw.split(',') if a.strip()])
    amenities_lower = amenities_raw.lower()

    # ── amenity flags ─────────────────────────────────────────────
    def has(terms):
        return int(any(t in amenities_lower for t in terms))

    has_pool            = has(['pool'])
    has_hot_tub         = has(['hot tub', 'jacuzzi'])
    has_parking         = has(['parking'])
    has_wifi            = has(['wifi', 'wi-fi', 'internet'])
    has_kitchen         = has(['kitchen'])
    has_washer          = has(['washer'])
    has_dryer           = has(['dryer'])
    has_air_conditioning= has(['air conditioning', 'ac'])
    has_workspace       = has(['workspace', 'dedicated workspace', 'desk'])
    allows_pets         = has(['pets allowed', 'pet friendly', 'pets'])
    has_beach_access    = has(['beach access', 'beachfront', 'waterfront'])
    has_gym             = has(['gym', 'exercise equipment'])
    has_elevator        = has(['elevator'])

    # ── engineered ratios ─────────────────────────────────────────
    guests_per_bedroom = guests / bedrooms
    beds_per_bedroom   = beds / bedrooms
    baths_per_guest    = baths / max(guests, 1)

    num_reviews = float(data.get('num_reviews', 0) or 0)
    rating_overall = float(data.get('rating_overall', 4.5) or 4.5)
    log_num_reviews       = np.log1p(num_reviews)
    ratings_x_log_reviews = rating_overall * log_num_reviews

    row = {
        # categorical
        'listing_type':            data.get('listing_type', 'Apartment'),
        'room_type':               data.get('room_type', 'Entire home/apt'),
        'superhost':               str(float(data.get('superhost', 0) or 0)),
        'professional_management': str(float(data.get('professional_management', 0) or 0)),
        'cancellation_policy':     data.get('cancellation_policy', 'Moderate'),
        'country':                 data.get('country', ''),
        'state':                   data.get('state', ''),
        'city':                    data.get('city', ''),
        'Region':                  data.get('region', 'North America'),

        # numeric
        'photos_count':            float(data.get('photos_count', 10) or 10),
        'latitude':                float(data.get('latitude', 0) or 0),
        'longitude':               float(data.get('longitude', 0) or 0),
        'guests':                  guests,
        'bedrooms':                bedrooms,
        'beds':                    beds,
        'baths':                   baths,
        'amenities_count':         amenities_count,
        'has_pool':                has_pool,
        'has_hot_tub':             has_hot_tub,
        'has_parking':             has_parking,
        'has_wifi':                has_wifi,
        'has_kitchen':             has_kitchen,
        'has_washer':              has_washer,
        'has_dryer':               has_dryer,
        'has_air_conditioning':    has_air_conditioning,
        'has_workspace':           has_workspace,
        'allows_pets':             allows_pets,
        'has_beach_access':        has_beach_access,
        'has_gym':                 has_gym,
        'has_elevator':            has_elevator,
        'guests_per_bedroom':      guests_per_bedroom,
        'beds_per_bedroom':        beds_per_bedroom,
        'baths_per_guest':         baths_per_guest,
        'min_nights':              float(data.get('min_nights', 2) or 2),
        'cleaning_fee':            float(data.get('cleaning_fee', 0) or 0),
        'extra_guest_fee':         float(data.get('extra_guest_fee', 0) or 0),
        'num_reviews':             num_reviews,
        'rating_overall':          rating_overall,
        'rating_accuracy':         float(data.get('rating_accuracy', rating_overall) or rating_overall),
        'rating_checkin':          float(data.get('rating_checkin', rating_overall) or rating_overall),
        'rating_cleanliness':      float(data.get('rating_cleanliness', rating_overall) or rating_overall),
        'rating_communication':    float(data.get('rating_communication', rating_overall) or rating_overall),
        'rating_location':         float(data.get('rating_location', rating_overall) or rating_overall),
        'rating_value':            float(data.get('rating_value', rating_overall) or rating_overall),
        'log_num_reviews':         log_num_reviews,
        'ratings_x_log_reviews':   ratings_x_log_reviews,
    }

    # Column order must exactly match features list from notebook
    feature_order = [
        "listing_type", "room_type", "photos_count", "superhost", "latitude", "longitude", "guests",
        "bedrooms", "beds", "baths", "amenities_count", "has_pool", "has_hot_tub", "has_parking",
        "has_wifi", "has_kitchen", "has_washer", "has_dryer", "has_air_conditioning", "has_workspace",
        "allows_pets", "has_beach_access", "has_gym", "has_elevator", "guests_per_bedroom",
        "beds_per_bedroom", "baths_per_guest", "professional_management", "min_nights",
        "cancellation_policy", "cleaning_fee", "extra_guest_fee", "num_reviews", "rating_overall",
        "rating_accuracy", "rating_checkin", "rating_cleanliness", "rating_communication",
        "rating_location", "rating_value", "log_num_reviews", "ratings_x_log_reviews",
        "country", "state", "city", "Region"
    ]

    return pd.DataFrame([row])[feature_order]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = build_dataframe(data)
        prediction = model.predict(df)[0]
        return jsonify({'amount': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/scrape', methods=['POST'])
def scrape():
    url = request.json.get('url', '')
    match = re.search(r'/rooms/(\d+)', url)
    if not match:
        return jsonify({'error': 'Not a valid Airbnb listing URL'})

    listing_id = match.group(1)
    headers = {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        resp = requests.get(f'https://www.airbnb.com/rooms/{listing_id}',
                            headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')

        data = {}
        for s in soup.find_all('script'):
            if s.string and 'homePDP' in (s.string or ''):
                import json
                json_match = re.search(r'\{.*\}', s.string, re.DOTALL)
                if json_match:
                    try:
                        raw = json.loads(json_match.group(0))
                        sections = (raw
                            .get('niobeMinimalClientData', [[]])[0][1]
                            .get('data', {})
                            .get('presentation', {})
                            .get('stayProductDetailPage', {})
                            .get('sections', {})
                            .get('metadata', {})
                            .get('loggingContext', {})
                            .get('eventDataLogging', {}))
                        data['bedrooms']       = sections.get('numBedrooms', '')
                        data['baths']          = sections.get('numBathrooms', '')
                        data['guests']         = sections.get('personCapacity', '')
                        data['city']           = sections.get('city', '')
                        data['country']        = sections.get('country', '')
                        data['rating_overall'] = sections.get('avgRating', '')
                        data['num_reviews']    = sections.get('reviewCount', '')
                    except Exception:
                        pass
                break

        return jsonify({
            'room_type':               data.get('room_type', ''),
            'listing_type':            data.get('listing_type', ''),
            'cancellation_policy':     data.get('cancellation_policy', ''),
            'bedrooms':                data.get('bedrooms', ''),
            'beds':                    data.get('beds', ''),
            'baths':                   data.get('baths', ''),
            'guests':                  data.get('guests', ''),
            'region':                  data.get('region', ''),
            'country':                 data.get('country', ''),
            'city':                    data.get('city', ''),
            'rating_overall':          data.get('rating_overall', ''),
            'rating_cleanliness':      data.get('rating_cleanliness', ''),
            'rating_location':         data.get('rating_location', ''),
            'rating_value':            data.get('rating_value', ''),
            'num_reviews':             data.get('num_reviews', ''),
            'photos_count':            data.get('photos_count', ''),
            'cleaning_fee':            data.get('cleaning_fee', ''),
            'extra_guest_fee':         data.get('extra_guest_fee', ''),
            'min_nights':              data.get('min_nights', ''),
            'amenities_count':         data.get('amenities_count', ''),
            'superhost':               data.get('superhost', False),
            'instant_book':            data.get('instant_book', False),
            'professional_management': data.get('professional_management', False),
        })

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Airbnb took too long to respond'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=False)