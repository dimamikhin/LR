from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from __future__ import annotations
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
            f"status={self.species!r},"
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
        classification: Optional[str] = None
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
        self.species = self.classification

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
            f"status={self.species!r},"
            f"classification={self.classification!r}"
            f")"
        )


class UnknownClient(Client):
    pass


class ClassifiedClient(Client):
    def __init__(self, classification: str, client: UnknownClient) -> None:
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
    def __init__(self, max_depth: int, min_sample_size: int, training: "TrainingData") -> None:
        self.max_depth = max_depth
        self.min_leaf_size = min_sample_size
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

        classifier = DecisionTreeClassifier()
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


    def classify(self, parameter: Hyperparameter, client: Client) -> Client:

        classification = parameter.classify(client)
        client.classify(classification)
        return client

    @staticmethod
    def get_list_clients(clients: list[Client]) -> list:
        return [
            [
                client.seniority,
                client.home,
                client.age,
                client.marital,
                client.records,
                client.expenses,
                client.assets,
                client.amount,
                client.price
            ]
            for client in clients
        ]

    @staticmethod
    def get_statuses_clients(clients: list[KnownClient]) -> list:
        return [client.status for client in clients]
