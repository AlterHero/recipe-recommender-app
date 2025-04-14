import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_recipe_data():
    df_recipes = pd.read_csv("data/13k-recipes.csv", usecols=['Title', 'Ingredients'])
    df_recipes.dropna(inplace=True)
    df_recipes['Ingredients'] = df_recipes['Ingredients'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    df_recipes = df_recipes[df_recipes['Ingredients'].apply(lambda x: len(x) > 0)]
    return df_recipes

@st.cache_resource
def train_recommender(df_input):
    mlb_encoder = MultiLabelBinarizer()
    X = mlb_encoder.fit_transform(df_input['Ingredients'])
    y = df_input['Title']
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model, mlb_encoder

def recommend_titles(ingredients, rf_model, mlb_encoder, df_source, top_n=5):
    vec_input = mlb_encoder.transform([ingredients])
    proba = rf_model.predict_proba(vec_input)[0]
    best_indices = proba.argsort()[-top_n:][::-1]
    top_recipes = rf_model.classes_[best_indices]
    return df_source[df_source['Title'].isin(top_recipes)][['Title']].drop_duplicates()

# Streamlit Web UI
st.title("🍽️ Рекомендатор рецептов по ингредиентам")
st.markdown("Введите список продуктов через запятую:")

input_ingredients = st.text_input("Ингредиенты", "chicken, onion, garlic")

if st.button("Показать рецепты"):
    recipes_df = load_recipe_data()
    model_rf, encoder_mlb = train_recommender(recipes_df)
    ingredients_list = [item.strip().lower() for item in input_ingredients.split(',')]
    result_df = recommend_titles(ingredients_list, model_rf, encoder_mlb, recipes_df)
    st.subheader("🔍 Найденные рецепты:")
    st.dataframe(result_df)