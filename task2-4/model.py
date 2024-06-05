from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

import datetime
from typing import (
    Optional,
    Union,
    Iterable
)

import weakref


class Client:

     def __init__(
        self,
        seniority: int,
        home: int,
        age: int,
        marital: int,
        records: int,
        expenses: int,
        assets: int,
        amount: int,
        price: int
    ) -> None:
        self.seniority = seniority
        self.home = home
        self.age = age
        self.marital = marital
        self.records = records
        self.expenses = expenses
        self.assets = assets
        self.amount = amount
        self.price = price

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"seniority={self.seniority}"
            f"home={self.home}"
            f"age={self.age}"
            f"marital={self.marital}"
            f"records={self.records}"
            f"expenses={self.assets}"
            f"amount={self.amount}"
            f"price={self.price}"
            f")"
        )


class KnownClient(Client):
    def __init__(self,
        status: int,
        seniority: int,
        home: int,
        age: int,
        marital: int,
        records: int,
        expenses: int,
        assets: int,
        amount: int,
        price: int
    ) -> None:
        super().__init__(
            seniority=seniority,
            home=home,
            age=age,
            marital=marital,
            records=records,
            expenses=expenses,
            assets=assets,
            amount=amount,
            price=price
        )
        self.status = status

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"seniority={self.seniority},"
            f"home={self.home},"
            f"age={self.age},"
            f"marital={self.marital},"
            f"records={self.records},"
            f"expenses={self.assets},"
            f"amount={self.amount},"
            f"price={self.price},"
            f"status={self.status!r},"
            f")"
        )


class TrainingKnownClient(KnownClient):
    pass


class TestingKnownClient(KnownClient):
    def __init__(self,
        status: int,
        seniority: int,
        home: int,
        age: int,
        marital: int,
        records: int,
        expenses: int,
        assets: int,
        amount: int,
        price: int,
        classification: Optional[int] = None
    ) -> None:
        super().__init__(
            status,
            seniority,
            home,
            age,
            marital,
            records,
            expenses,
            assets,
            amount,
            price
        )
        self.classification = classification


    def mathces(self) -> bool:
        self.status = self.classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"seniority={self.seniority},"
            f"home={self.home},"
            f"age={self.age},"
            f"marital={self.marital},"
            f"records={self.records},"
            f"expenses={self.assets},"
            f"amount={self.amount},"
            f"price={self.price},"
            f"status={self.status!r},"
            f"classification={self.classification!r}"
            f")"
        )


class UnknownClient(Client):
    pass


class ClassifiedClient(Client):
    def __init__(self, classification: int, client: UnknownClient) -> None:
        super().__init__(
            seniority=client.seniority,
            home=client.home,
            age=client.age,
            marital=client.marital,
            records=client.records,
            expenses=client.expenses,
            assets=client.assets,
            amount=client.amount,
            price=client.price
        )
        self.classification = classification

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"seniority={self.seniority},"
            f"home={self.home},"
            f"age={self.age},"
            f"marital={self.marital},"
            f"records={self.records},"
            f"expenses={self.assets},"
            f"amount={self.amount},"
            f"price={self.price},"
            f"classification={self.classification!r}"
            f")"
        )


class Hyperparameter:
    def __init__(self, max_depth: int, min_samples_split: int, training: "TrainingData") -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float


    def test(self) -> None:
        trainingData: Optional["TrainingData"] = self.data()
        if not trainingData:
            raise RuntimeError("Broken Waek Reference")
        test_data = trainingData.testing
        x_test = TrainingData.get_list_clients(test_data)
        y_test = TrainingData.get_statuses_clients(test_data)
        y_predict = self.classify_list(x_test)
        self.quality = roc_auc_score(y_test, y_predict)
        for i in range(len(y_predict)):
            test_data[i].classification = y_predict[i]


    def classify_list(self, clients: list[Union[UnknownClient, TestingKnownClient]]) -> list:
        training_data = self.data
        if not training_data:
            raise RuntimeError("No training object")
        x_predict = TrainingData.get_list_clients(clients)
        x_train = TrainingData.get_list_clients(training_data)
        y_train = TrainingData.get_statuses_clients(training_data)

        classifier = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        classifier = classifier.fit(x_train, y_train)
        return classifier.predict(x_predict).tolist()


class TrainingData:
    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[TrainingKnownClient] = []
        self.testing: list[TestingKnownClient] = []
        self.tuning: list[Hyperparameter] = []

    def load(self, raw_data_soruce: Iterable[dict[str, str]]) -> None:

        for n, row in enumerate(raw_data_soruce):
            client = Client(
                seniority = int(row["seniority"]),
                home = int(row["home"]),
                age = int(row["age"]),
                marital = int(row["marital"]),
                records = int(row["records"]),
                expenses = int(row["expenses"]),
                assets = int(row["assets"]),
                amount = int(row["amount"]),
                price = int(row["price"]),
                status = row["status"]
            )
            if n % 5 == 0:
                self.testing.append(client)
            else:
                self.training.append(client)
        self.uploaded = datetime.date.now(tz=datetime.timezone.utc)


    def test(self, parameter: Hyperparameter) -> None:

        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)



    def classify(self, parameter: Hyperparameter, client: UnknownClient) -> ClassifiedClient:


        return ClassifiedClient(classification=parameter.classify_list(TrainingData.get_client_as_list(client)), client=client)


KnownClient = """
>>> x = Sample(1, 1, 1, 1, 1, 1, 1, 1, 1)
>>> x
UnknownSample(seniority=1, home=1, age=1, marital=1, records=1, expenses=1, assets=1, amount=1, price=1, status=1)
"""

test_UnknownClient = """
>>> u = Client(2,2,2,2,2,2,2,2,2)
>>> u
Client(seniority=2, home=2, age=2, marital=2, records=2, expenses=2, assets=2, amount=2, price=2)
"""
