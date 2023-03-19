import openai
import pandas as pd
import re
import csv
import os

openai.api_key = os.environ["OPENAI_API_KEY"]  # Stored as an environment variable for security reasons


def read_reviews_from_file(filename):
    """ Open the file and convert its content into
     a python list of table rows for each review """
    try:
        with open(f"{filename}.csv", 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            review_data = [row for row in reader]
            return review_data
    except Exception as e:
        print(f"Error reading {filename}.csv file:", e)
        return None


def rate_reviews(review_data):
    """ Rate each review 1-10 using the OpenAI API
    and store the result in a new 'rate' column """
    try:
        for row in review_data:
            review = row["review text"]  # Take the review text from the row
            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=f"Rate the tone of the following review on a scale from 1 to 10, "
                       f"with 10 being the most positive: {review}"
                       f"and giving only the rate number for each of the reviews in your response.",
                max_tokens=50,  # Indicates how 'long' the response text will be in 'tokens' (~words)
                n=1,  # Requests only one response for the given prompt
                stop=None,  # Asks the OpenAPI to only stop generating text when 'max_tokens' is reached
                temperature=0.5,  # Sets the 'creativity' level for the response
                seed=1,  # Provides consistency in multiple-time rating of the same dataset
            )
            rating_str = re.search('\d+', response.choices[0].text).group()  # Get rid of everything but the number
            row["rate"] = int(rating_str)
    except openai.error.OpenAIError as e:
        print("OpenAI API error:", e)
    except Exception as e:
        print("Error processing reviews:", e)
    return review_data


def save_analyzed_reviews(review_data, filename):
    """ Convert the review data to a Pandas DataFrame and sort it
    by the 'rate' column. Then save the sorted data as a new CSV file """
    try:
        df = pd.DataFrame(review_data)
        df_sorted = df.sort_values(by=["rate"], ascending=False)
        df_sorted['rate'] = df_sorted['rate'].astype(int)
        df_sorted.to_csv(f"{filename}_analyzed.csv", index=False)
    except Exception as e:
        print(f"Error writing {filename}_analyzed.csv file:", e)


def main():
    user_filename = input("\n"
                          "Put the reviews file to the script folder and provide its name "
                          "here (the file should have a .csv extension).\n\n"
                          "Don't type the '.csv' part, just type the name of the file: ")
    reviews_data = read_reviews_from_file(user_filename)
    rated_reviews = rate_reviews(reviews_data)
    save_analyzed_reviews(rated_reviews, user_filename)
    print("\n"
          "Thank you! If no errors occured while running the script, your file is now ready.\n"
          "If you see any errors that occured while the script was running, please fix them "
          "and re-run the script. ")


if __name__ == "__main__":
    main()
