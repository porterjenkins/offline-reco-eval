from datetime import datetime
from typing import Optional


class DisplayState(object):

    def __init__(self, disp_id: str, prod_quantity: Optional[dict] = None, max_slots: Optional[int] = None, timestamp: Optional[datetime] = None):
        self.disp_id = disp_id
        self.max_slots = max_slots
        self.ts = timestamp
        if prod_quantity is not None:
            self.prods = set(prod_quantity.keys())
        self.quantities = prod_quantity


    def get_prod_quantites(self):
        return self.quantities


    def set_time(self, ts: datetime):
        self.ts = ts

    def set_max_slots(self, max_slots: int):
        self.max_slots = max_slots

    def __str__(self):
        return str(self.ts) + ": " + str(self.prods)

    def __len__(self):
        return len(self.prods)
