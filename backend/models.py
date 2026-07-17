from sqlalchemy import Column, Integer, String, Float
from backend.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    phone = Column(String, nullable=True)
    # RFM metrics for the model
    frequency = Column(Float, default=0.0)
    monetary = Column(Float, default=0.0)
    tenure = Column(Float, default=0.0)
