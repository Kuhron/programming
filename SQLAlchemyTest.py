import os
import random
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()  # subclass this to make records

# tutorial: https://docs.sqlalchemy.org/en/13/orm/tutorial.html

class MyObject:
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2
        self.name = str(arg1)[::-1]
        self.address = MyObject.transform_str(arg2)  # see if this stuff ends up in the database

    @staticmethod
    def transform_str(s):
        s = str(s)[::-1]
        s += "_" + "".join(random.choice("aoeuidhtns") for _ in range(10))
        return s

    def to_record(self):
        return MyObjectRecord(name=self.name, address=self.address)


class MyObjectRecord(Base):
    # class variables tell SQLAlchemy how to put this object in the db
    __tablename__ = "testtable"
    name = Column(String(length=5), primary_key=True)
    address = Column(String(length=5))

    def __repr__(self):  # db doesn't care about this, it's only for my own debugging
        return "MyObjectRecord : name={} address={}".format(self.name, self.address)  # I guess the class variables are turned into instance variables somewhere in the parent Base class?
    

if __name__ == "__main__":
    db_dir = os.path.dirname(os.path.abspath(__file__))
    db_filename = "sqlalchemytest.db"
    db_fp = "sqlite:///" + os.path.join(db_dir, db_filename)  # if you put only two slashes it will cause permission error
    print("db_fp is {}".format(db_fp))
    
    engine = create_engine(db_fp, echo=True)  # does not actually create the db file yet
    # echo=True will also log all statements to stdout
    Base.metadata.create_all(engine)  # actually create the table
    
    Session = sessionmaker(bind=engine)  # creates a class
    
    # create a Session
    session = Session()
    
    # work with sess
    my_object = MyObject('foo', 'bar')
    my_object_record = my_object.to_record()  # you don't have to do this, can just instantiate the record directly, but this kind of flow might be useful when dealing with complicated objects that have a bunch of methods, but whose db records only need a few fields
    print("record is: {}".format(my_object_record))

    exists = session.query(MyObjectRecord).filter_by(name=my_object_record.name).first()
    if not exists:
        session.add(my_object_record)
        session.commit()

    retrieved_object = session.query(MyObjectRecord).filter_by(name="oof").first()
    print("retrieved {}".format(retrieved_object))


