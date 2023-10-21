from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
)
from metaflow.cards import Table, Markdown, Artifact

def labeling_function(row):
    """
    A function to derive labels from the user's review data.
    This could use many variables, or just one.
    In supervised learning scenarios, this is a very important part of determining what the machine learns!

    A subset of variables in the e-commerce fashion review dataset to consider for labels you could use in ML tasks include:
        # rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
        # recommended_ind: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not recommended.
        # positive_feedback_count: Positive Integer documenting the number of other customers who found this review positive.

    In this case, we are doing sentiment analysis.
    To keep things simple, we use the rating only, and return a binary positive or negative sentiment score based on an arbitrarty cutoff.
    """

    if row["rating"] >= 4:
        return 1
    else:
        return 0


class GoodFirstModelNLPFlow(FlowSpec):
    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter("split-sz", default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile("data", default="../data/Womens Clothing E-Commerce Reviews.csv")

    @step
    def start(self):
        # Step-level dependencies are loaded within a Step, instead of loading them
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df["review_text"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({"label": labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.log_reg_model)

    @step
    def log_reg_model(self):

        ### TODO: Fit and score a logistic regression model on the data, log the acc and rocauc as artifacts.
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.feature_extraction.text import CountVectorizer

        # TF-IDF vectorization:
        vectorizer = TfidfVectorizer(max_features=5000)
        
        model = LogisticRegression()
        model.fit(vectorizer.fit_transform(self.traindf['review']), self.traindf['label'])
        predictions = model.predict(vectorizer.transform(self.valdf['review']))
        probability = model.predict_proba(vectorizer.transform(self.valdf['review']))[:, 1]
        self.log_reg_acc = accuracy_score(self.valdf['label'], predictions)
        self.log_reg_rocauc = roc_auc_score(self.valdf['label'], probability)

        self.next(self.end)

    @card(
        type="corise"
    )  # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        msg = "Logistic Regression Accuracy: {}\nLogistic Regression AUC: {}"
        print(msg.format(round(self.log_reg_acc, 3), round(self.log_reg_rocauc, 3)))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.log_reg_acc))

        current.card.append(Markdown("## Examples of False Positives"))
        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0.
        false_positive_mask = (self.valdf['label'] == 0) & (self.predictions == 1)
        false_positive_df = self.valdf[false_positive_mask]
        # TODO: display the false_positives dataframe using metaflow.cards
        # Documentation: https://docs.metaflow.org/api/cards#table
        current.card.append(
            Table.from_dataframe(false_positive_df.head(10))
        )

        current.card.append(Markdown("## Examples of False Negatives"))
        # TODO: compute the false negative predictions where the baseline is 0 and the valdf label is 1.
        false_negative_mask = (self.valdf['label'] == 1) & (self.predictions == 0)
        false_negative_df = self.valdf[false_negative_mask]
        # TODO: display the false_negatives dataframe using metaflow.cards
        current.card.append(
            Table.from_dataframe(false_negative_df.head(10))
        )


if __name__ == "__main__":
    GoodFirstModelNLPFlow()
