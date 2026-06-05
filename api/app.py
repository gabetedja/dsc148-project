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

    bedrooms = max(float(data.get('bedrooms', 1) or 1), 1)
    guests   = float(data.get('guests', 2) or 2)
    beds     = float(data.get('beds', bedrooms) or bedrooms)
    baths    = float(data.get('baths', 1) or 1)

    amenities_raw = str(data.get('amenities_raw', '') or '')
    amenities_count = int(data.get('amenities_count', 0) or 0)
    if amenities_raw:
        amenities_count = len([a for a in amenities_raw.split(',') if a.strip()])
    amenities_lower = amenities_raw.lower()

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

    # from nb
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

    try:
        resp = requests.get(f'https://www.airbnb.com/rooms/{listing_id}', timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')

        ld = {}
        for tag in soup.find_all('script', type='application/ld+json'):
            try:
                obj = json.loads(tag.string or '')
                if obj.get('@type') == 'VacationRental':
                    ld = obj
                    break
            except Exception:
                pass

        rating_overall  = ld.get('aggregateRating', {}).get('ratingValue', '')
        num_reviews     = ld.get('aggregateRating', {}).get('ratingCount', '')
        latitude        = ld.get('latitude', '')
        longitude       = ld.get('longitude', '')
        city            = ld.get('address', {}).get('addressLocality', '')
        guests_ld       = ld.get('containsPlace', {}).get('occupancy', {}).get('value', '')
        photos_count    = len(ld.get('image', []))

        bedrooms = beds = baths = guests = ''
        overview_el = soup.find('ol', class_=lambda c: c and 'lgx66tx' in c)
        if overview_el:
            items = [li.get_text(strip=True) for li in overview_el.find_all('li')]
            for item in items:
                item_lower = item.lower()
                m = re.search(r'(\d+)\s+guest', item_lower)
                if m: guests = m.group(1)
                m = re.search(r'(\d+)\s+bedroom', item_lower)
                if m: bedrooms = m.group(1)
                m = re.search(r'(\d+)\s+bed', item_lower)
                if m: beds = m.group(1)
                m = re.search(r'([\d.]+)\s+bath', item_lower)
                if m: baths = m.group(1)

        # Fall back to JSON-LD guests if not found in overview
        if not guests and guests_ld:
            guests = guests_ld

        room_type = ''
        listing_type = ''
        h2_overview = soup.find('h2', string=re.compile(r'Entire|Private room|Shared room|Hotel', re.I))
        if h2_overview:
            text = h2_overview.get_text(strip=True).lower()
            if 'entire' in text:
                room_type = 'Entire home/apt'
            elif 'private room' in text:
                room_type = 'Private room'
            elif 'shared room' in text:
                room_type = 'Shared room'
            elif 'hotel' in text:
                room_type = 'Hotel room'

            # listing type from same heading 
            type_match = re.search(r'entire\s+(\w+)', text)
            if type_match:
                raw = type_match.group(1).lower()
                type_map = {
                    'house': 'House', 'apartment': 'Apartment', 'condo': 'Condo',
                    'loft': 'Loft', 'villa': 'Villa', 'cabin': 'Cabin',
                    'studio': 'Studio', 'guesthouse': 'Other', 'townhouse': 'Townhouse',
                }
                listing_type = type_map.get(raw, 'Other')

        country = state = ''
        location_el = soup.find(class_=lambda c: c and 's1u3608j' in (c or ''))
        if location_el:
            loc_text = location_el.get_text(strip=True)  
            parts = [p.strip() for p in loc_text.split(',')]
            if len(parts) >= 3:
                city = city or parts[0]
                state = parts[1]
                country = parts[2]
            elif len(parts) == 2:
                city = city or parts[0]
                country = parts[1]

        #amenities stuff
        amenities_raw = ''
        amenity_items = []
        amenities_section = soup.find(attrs={'data-section-id': 'AMENITIES_DEFAULT'})
        if amenities_section:
            for div in amenities_section.find_all('div', class_=lambda c: c and 'i3qbefn' in (c or '')):
                text = div.get_text(separator=' ', strip=True)
                # strip out icon text / duplication
                clean = re.sub(r'\s+', ' ', text).strip()
                if clean and len(clean) < 60:
                    amenity_items.append(clean)

        # Also grab count from "Show all N amenities" button
        amenities_count = len(amenity_items)
        btn_text = soup.find(string=re.compile(r'Show all \d+ amenities'))
        if btn_text:
            m = re.search(r'(\d+)', btn_text)
            if m:
                amenities_count = int(m.group(1))

        amenities_raw = ', '.join(amenity_items) if amenity_items else ''

        def extract_subrating(label):
            heading = soup.find(string=re.compile(
                rf'Rated ([\d.]+) out of 5 stars for {label}', re.I))
            if heading:
                m = re.search(r'Rated ([\d.]+)', heading, re.I)
                if m: return m.group(1)
            return ''

        rating_cleanliness   = extract_subrating('cleanliness')
        rating_accuracy      = extract_subrating('accuracy')
        rating_checkin       = extract_subrating('check-in')
        rating_communication = extract_subrating('communication')
        rating_location_r    = extract_subrating('location')
        rating_value         = extract_subrating('value')

        superhost = bool(soup.find(string=re.compile(r'Superhost', re.I)))

        region_map = {
            'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
            'United Kingdom': 'Europe', 'France': 'Europe', 'Germany': 'Europe',
            'Italy': 'Europe', 'Spain': 'Europe', 'Portugal': 'Europe',
            'Japan': 'Asia-Pacific', 'Australia': 'Asia-Pacific', 'Thailand': 'Asia-Pacific',
            'Brazil': 'Latin America/Caribbean', 'Colombia': 'Latin America/Caribbean',
        }
        region = region_map.get(country.strip(), '')

        return jsonify({
            'room_type':               room_type,
            'listing_type':            listing_type,
            'cancellation_policy':     '',
            'bedrooms':                bedrooms,
            'beds':                    beds,
            'baths':                   baths,
            'guests':                  guests,
            'region':                  region,
            'country':                 country,
            'state':                   state,
            'city':                    city,
            'latitude':                latitude,
            'longitude':               longitude,
            'rating_overall':          rating_overall,
            'rating_cleanliness':      rating_cleanliness,
            'rating_accuracy':         rating_accuracy,
            'rating_checkin':          rating_checkin,
            'rating_communication':    rating_communication,
            'rating_location':         rating_location_r,
            'rating_value':            rating_value,
            'num_reviews':             num_reviews,
            'photos_count':            photos_count,
            'amenities_count':         amenities_count,
            'amenities_raw':           amenities_raw,
            'superhost':               superhost,
            'instant_book':            False,
            'professional_management': False,
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