import pandas as pd

from preprocessing import match_preference, is_deal_size_in_range, preprocess_rfp_features, preprocess_lenders_features
from sklearn.metrics.pairwise import cosine_similarity

def get_top_n_similar_rfps(
    rfp_id, 
    X, 
    top_n=5, 
    df_keys=None, 
    ensure_lender=False, 
    method='cosine_similarity'
):
    if method != 'cosine_similarity':
        raise NotImplementedError("Only cosine_similarity is currently supported.")
    
    if rfp_id not in X.index:
        raise ValueError(f"RFP ID '{rfp_id}' not found in the dataset.")

    # Compute similarity
    similarity_matrix = cosine_similarity(X)
    similarity_df = pd.DataFrame(similarity_matrix, index=X.index, columns=X.index)
    similarities = similarity_df.loc[rfp_id].drop(rfp_id).sort_values(ascending=False)

    # If not filtering by lender, return top_n directly
    if not ensure_lender or df_keys is None:
        return similarities.head(top_n)

    # Keep only the first RFP per unique awarded_lender_id
    df_sim = similarities.reset_index()
    df_sim.columns = ['rfp_id', 'similarity']
    df_sim = df_sim.merge(df_keys[['rfp_id', 'awarded_lender_id']], on='rfp_id', how='left')
    df_sim = df_sim[df_sim['awarded_lender_id'].notna()]
    df_unique = df_sim.drop_duplicates(subset='awarded_lender_id', keep='first')

    # Return top_n most similar
    return df_unique.set_index('rfp_id')['similarity'].head(top_n)

def sort_suggested_lenders_to_rfp(rfp_row, awarded_lenders_ids, df_cleaned_lender_preferences):
    # Filter suggested lenders
    df_suggested_lenders = df_cleaned_lender_preferences[df_cleaned_lender_preferences['lender_id'].isin(awarded_lenders_ids)].copy()

    # Calculate matches (Region > deal_size > loan_type > industry_sector)
    df_suggested_lenders['region_match'] = df_suggested_lenders.apply(
        lambda row: match_preference(row['preferred_regions'], rfp_row['region']), axis=1)
    df_suggested_lenders['deal_size_in_range'] = df_suggested_lenders.apply(
        lambda row: is_deal_size_in_range(rfp_row['deal_size_usd'], row['preferred_deal_size_min_usd'], row['preferred_deal_size_max_usd']), axis=1)
    df_suggested_lenders['loan_type_match'] = df_suggested_lenders.apply(
        lambda row: match_preference(row['preferred_loan_types'], rfp_row['loan_type_requested']), axis=1)
    df_suggested_lenders['industry_match'] = df_suggested_lenders.apply(
        lambda row: match_preference(row['preferred_industries'], rfp_row['industry_sector']), axis=1)
    df_suggested_lenders['total_matches'] = (
        df_suggested_lenders['region_match'].astype(int) +
        df_suggested_lenders['deal_size_in_range'].astype(int) +
        df_suggested_lenders['loan_type_match'].astype(int) +
        df_suggested_lenders['industry_match'].astype(int)
    )

    # Create a tuple key for sorting based on priority (region > industry > loan_type > deal_size)
    df_suggested_lenders['sort_key'] = df_suggested_lenders.apply(
        lambda row: (
            row['total_matches'],
            int(row['region_match']), 
            int(row['deal_size_in_range']),
            int(row['loan_type_match']), 
            int(row['industry_match']) 
        ), axis=1
    )

    # Return sorted suggested lenders by number of matches and matches priority
    return df_suggested_lenders.sort_values(by='sort_key', ascending=False).drop(columns='sort_key')

def get_suggested_lenders_to_rfp(rfp_row, similar_rfps_list, df_rfp_lender, df_cleaned_lender_preferences):
    # Filter suggested lenders
    awarded_lenders_ids = df_rfp_lender[df_rfp_lender['rfp_id'].isin(similar_rfps_list)]['awarded_lender_id'].to_list()
    df_suggested_lenders = df_cleaned_lender_preferences[df_cleaned_lender_preferences['lender_id'].isin(awarded_lenders_ids)].copy()

    df_suggested_lenders = sort_suggested_lenders_to_rfp(rfp_row, 
                                  awarded_lenders_ids, 
                                  df_cleaned_lender_preferences)
    
    return df_suggested_lenders

def similar_rfp_profile_recommendation(rfp_id, df_cleaned_historical_rfps, df_cleaned_lender_preferences, top_n=10):
    rfp_row = df_cleaned_historical_rfps[df_cleaned_historical_rfps['rfp_id'] == rfp_id].iloc[0]

    # prepare data
    df_rfp_lender = df_cleaned_historical_rfps[['rfp_id', 'awarded_lender_id']]
    df_prep_historical_rfps = preprocess_rfp_features(df_cleaned_historical_rfps)

    # Get similar rfps
    similar_rfps = get_top_n_similar_rfps(
        rfp_id, 
        df_prep_historical_rfps, 
        top_n=top_n, 
        df_keys=df_rfp_lender,
        ensure_lender=True,
        method='cosine_similarity')

    # Get top suggestions based on similar rfps
    df_suggested_lenders = get_suggested_lenders_to_rfp(rfp_row, 
                                                        similar_rfps.index, 
                                                        df_rfp_lender, 
                                                        df_cleaned_lender_preferences)

    return df_suggested_lenders

def get_top_n_similar_lenders(
    lender_id,
    X_lenders,
    top_n=5,
    method='cosine_similarity'
):
    if method != 'cosine_similarity':
        raise NotImplementedError("Only cosine_similarity is currently supported.")

    if lender_id not in X_lenders.index:
        raise ValueError(f"Lender ID '{lender_id}' not found in the dataset.")

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(X_lenders)
    similarity_df = pd.DataFrame(similarity_matrix, index=X_lenders.index, columns=X_lenders.index)

    # Get top-N most similar lenders, excluding itself
    similarities = similarity_df.loc[lender_id].drop(lender_id).sort_values(ascending=False)

    return similarities.head(top_n)

# Get recommendations based on lenders similarities (Use the awarded_lender of the most similar rfp if the given rfp has no awarded_lender)
def similar_lenders_profile_recommendation(rfp_id, df_cleaned_historical_rfps, df_cleaned_lender_preferences, top_n=10):
    rfp_row = df_cleaned_historical_rfps[df_cleaned_historical_rfps['rfp_id'] == rfp_id].iloc[0]
    
    # prepare data
    df_rfp_lender = df_cleaned_historical_rfps[['rfp_id', 'awarded_lender_id']]
    df_prep_historical_rfps = preprocess_rfp_features(df_cleaned_historical_rfps)
    df_prep_lender_preferences = preprocess_lenders_features(df_cleaned_lender_preferences)

    # Get awarded lender
    rfp_row = df_cleaned_historical_rfps[df_cleaned_historical_rfps['rfp_id'] == rfp_id].iloc[0]
    lender_id = rfp_row['awarded_lender_id']
    if lender_id is None:
        similar_rfps = get_top_n_similar_rfps(
            rfp_id, 
            df_prep_historical_rfps, 
            top_n=1, 
            df_keys=df_rfp_lender,
            ensure_lender=True,
            method='cosine_similarity')
        rfp_id = similar_rfps.index[0]
        rfp_row = df_cleaned_historical_rfps[df_cleaned_historical_rfps['rfp_id'] == rfp_id].iloc[0]
        lender_id = rfp_row['awarded_lender_id']

    # Get recommendations (Similar lenders)
    similar_lenders = get_top_n_similar_lenders(
        lender_id,
        df_prep_lender_preferences,
        top_n=top_n
    )

    # Sort recommendations
    df_suggested_lenders = sort_suggested_lenders_to_rfp(rfp_row, 
                              similar_lenders.index, 
                              df_cleaned_lender_preferences)

    return df_suggested_lenders