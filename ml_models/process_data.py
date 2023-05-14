import pandas as pd
from datetime import datetime, date
from microservice.utils.path_to import path_to
from sklearn.feature_extraction import FeatureHasher

N_FEATURES_PROVINCE = 2**5
N_FEATURES_CATEGORY = 2**5

def process_category(category):
    fh = FeatureHasher(n_features=N_FEATURES_CATEGORY, input_type='string')
    return fh.fit_transform([[category]]).toarray()

def process_sex(fullname):
    return 1.0 if fullname.split(' ')[0][-1] == 'a' else 0.0

def process_city(city):
    fh = FeatureHasher(n_features=N_FEATURES_PROVINCE, input_type='string')
    return fh.fit_transform([[city]]).toarray()

def process_discount(discount):
    return discount / 100

if __name__ == "__main__":

    def mask_val(df, col, val):
        mask = df[col].values == val
        return df[mask]

    #########################

    print("START: READING")
    products = pd.read_json(path_to(__file__, 'data/preprocessed/products.jsonl'), lines=True)
    sessions = pd.read_json(path_to(__file__, 'data/preprocessed/sessions.jsonl'), lines=True)
    users = pd.read_json(path_to(__file__, 'data/preprocessed/users.jsonl'), lines=True)
    print("FINISH: READING")

    #########################

    def read_categories():
        print("START: CATEGORIES")
        fh = FeatureHasher(n_features=N_FEATURES_CATEGORY, input_type='string')
        categories = []
        categories_raw = []
        for id in sessions['product_id']:
            category = mask_val(products, 'product_id', id)['category_path']
            if len(category.values) == 0:
                categories.append([None])
                categories_raw.append(None)
            else:
                categories_raw.append(str(category.values[0]))
                categories.append([str(category.values[0])])
        categories_fh = fh.fit_transform(categories).toarray()
        print("FINISH: CATEGORIES")
        return categories_fh, categories_raw

    #########################

    def read_prices():
        print("START: PRICES")
        prices = []
        for id in sessions['product_id']:
            product_price = mask_val(products, 'product_id', id)['price']
            if len(product_price.values) == 0:
                prices.append(0)
            elif product_price.values[0] < 0 or product_price.values > 100000:
                prices.append(0)
            else:
                prices.append(product_price.values[0])
        print("FINISH: PRICES")
        return prices

    #########################

    def read_buy_ended():
        print("START: BUY_ENDED")
        idx = mask_val(sessions, 'event_type', "BUY_PRODUCT")
        sessions_buy = set(idx['session_id'])
        buy_ended = [1 if session_id in sessions_buy else 0 for session_id in sessions['session_id']]
        print("FINISH: BUY_ENDED")
        return buy_ended

    #########################

    def read_session_time():
        print("START: SESSION_TIME")
        session_time = []
        current_session = -1
        start_time = sessions['timestamp'][0].time()
        for index, row in sessions.iterrows():
            if row['session_id'] != current_session:
                current_session = row['session_id']
                start_time = row['timestamp'].time()
                current_time = row['timestamp'].time()
                session_time.append(datetime.combine(date.today(), current_time) - datetime.combine(date.today(), start_time))
            else:
                current_time = row['timestamp'].time()
                session_time.append(datetime.combine(date.today(), current_time) - datetime.combine(date.today(), start_time))
        print("FINISH: SESSION_TIME")
        return session_time

    #########################

    def read_sex():
        print("START: SEX")
        sex = []
        names = []
        for id in sessions['user_id']:
            fullname = mask_val(users, 'user_id', id)['name'].values[0] # type:ignore
            sex.append(process_sex(fullname))
            names.append(fullname)
        print("FINISH: SEX")
        return sex, names

    ################################

    def read_city():
        print("START: PROVINCE")

        fh = FeatureHasher(n_features=N_FEATURES_PROVINCE, input_type='string')
        cities = []
        cities_raw = []
        for id in sessions['user_id']:
            user = mask_val(users, 'user_id', id)
            if len(user) == 0:
                cities.append([None])
                cities_raw.append(None)
            else:
                city = user['city'].values[0]
                cities.append([city])
                cities_raw.append(city)
        print("FINISH: PROVINCE")
        cities_fh = fh.fit_transform(cities).toarray()
        return cities_fh, cities_raw

    ######################

    def columns_to_dict(prices, categories, categories_raw, session_time, sex, names, provinces, cities):
        print("START: COLUMNS")
        columns = {
            "price": prices,
            "session_time": session_time,
            "discount": sessions["offered_discount"],
            "sex": sex,
        }
        for i in range(categories.shape[1]):
            columns[f"c{i}"] = categories[:, i]
        for i in range(provinces.shape[1]):
            columns[f"p{i}"] = provinces[:, i]

        columns2 = {
            "price": prices,
            "session_time": session_time,
            "discount": sessions["offered_discount"],
            "name": names,
            "category": categories_raw,
            "city": cities
        }

        print("FINISH: COLUMNS")
        return columns, columns2

    ##########################

    def the_final_cut(columns, columns2, buy_ended):
        print("START: FINAL DATAFRAME")
        data = pd.DataFrame(data=columns)
        data["session_time"] = data["session_time"].dt.total_seconds()
        data['discount'] = process_discount(data['discount'])
        data['buy_ended'] = buy_ended
        data.drop(data.loc[data["session_time"] < 0].index, inplace=True)

        client_data = pd.DataFrame(data=columns2)
        client_data["session_time"] = client_data["session_time"].dt.total_seconds()
        client_data['buy_ended'] = buy_ended
        client_data.drop(client_data.loc[client_data["session_time"] < 0].index, inplace=True)

        print("FINISH: FINAL DATAFRAME")
        return data, client_data

    ###########################

    def dump(data, client_data):
        print("START: FINAL DUMP")
        data.to_json(path_to(__file__, 'data/processed/processed_data.jsonl'), orient="records", lines=True)
        client_data.to_json(path_to(__file__, 'data/processed/client_data.jsonl'), orient="records", lines=True)
        #data.to_csv(path_to(__file__, 'data/processed/processed_data.csv'), header=False, index=False)
        print("FINISH: FINAL DUMP")

    ###############################

    c, c2 = read_categories()
    p = read_prices()
    b = read_buy_ended()
    t = read_session_time()
    s, s2 = read_sex()
    v, v2 = read_city()
    d, d2 = columns_to_dict(p, c, c2, t, s, s2, v, v2)
    df, df2 = the_final_cut(d, d2, b)
    dump(df, df2)
