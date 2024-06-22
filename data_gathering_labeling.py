import os
import openpyxl
import yfinance
import matplotlib.pyplot as plt
import numpy as np
import csv

class FullDataPoint:
    def __init__(self, emp=None, pe=None, cape=None, dy=None, rho=None, mov=None, ir=None, rr=None, y02=None, y10=None, stp=None, cf=None, mg=None, rv=None, ed=None, un=None, gdp=None, m2=None, cpi=None, dil=None, yss=None, nyf=None, _au=None, _dxy=None, _lcp=None, _ty=None, _oil=None, _mkt=None, _va=None, _gr=None, snp=None, label=None):
        self.emp = emp
        self.pe = pe
        self.cape = cape
        self.dy = dy
        self.rho = rho
        self.mov = mov
        self.ir = ir
        self.rr = rr
        self.y02 = y02
        self.y10 = y10
        self.stp = stp
        self.cf = cf
        self.mg = mg
        self.rv = rv
        self.ed = ed
        self.un = un
        self.gdp = gdp
        self.m2 = m2
        self.cpi = cpi
        self.dil = dil
        self.yss = yss
        self.nyf = nyf
        self._au = _au
        self._dxy = _dxy
        self._lcp = _lcp
        self._ty = _ty
        self._oil = _oil
        self._mkt = _mkt
        self._va = _va
        self._gr = _gr
        self.snp = snp
        self.label = label

    def __repr__(self) -> str:
        return f"{self.emp}, {self._oil}, {self.label}"

class DataPointStock:
    def __init__(self, date, value, volume, dividends = 0, stock_splits = 0):
        self.date = date
        self.value = value
        self.volume = volume
        self.dividends = dividends
        self.stock_splits = stock_splits

    def __repr__(self) -> str:
        return f"{self.date, self.value, self.volume, self.dividends, self.stock_splits}"

    def get_date(self):
        return self.date
    
    def set_date(self, date):
        self.date = date
    
    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def get_volume(self):
        return self.volume
    
    def set_volume(self, volume):
        self.volume = volume
    
    def get_dividends(self):
        return self.dividends
    
    def set_dividends(self, dividends):
        self.dividends = dividends
    
    def get_stock_splits(self):
        return self.stock_splits
    
    def set_stock_splits(self, stock_splits):
        self.stock_splits = stock_splits

class DataPoint:
    def __init__(self, date, value):
        self.date = date
        self.value = value

    def __repr__(self) -> str:
        return f"{self.date, self.value}"

    def get_date(self):
        return self.date
    
    def set_date(self, date):
        self.date = date
    
    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

class Data:

    def __init__(self, emp=None, pe=None, cape=None, dy=None, rho=None, mov=None, ir=None, rr=None, y02=None, y10=None, stp=None, cf=None, mg=None, rv=None, ed=None, un=None, gdp=None, m2=None, cpi=None, dil=None, yss=None, nyf=None, _au=None, _dxy=None, _lcp=None, _ty=None, _oil=None, _mkt=None, _va=None, _gr=None, snp=None, label=None):
        self.emp = emp
        self.pe = pe
        self.cape = cape
        self.dy = dy
        self.rho = rho
        self.mov = mov
        self.ir = ir
        self.rr = rr
        self.y02 = y02
        self.y10 = y10
        self.stp = stp
        self.cf = cf
        self.mg = mg
        self.rv = rv
        self.ed = ed
        self.un = un
        self.gdp = gdp
        self.m2 = m2
        self.cpi = cpi
        self.dil = dil
        self.yss = yss
        self.nyf = nyf
        self._au = _au
        self._dxy = _dxy
        self._lcp = _lcp
        self._ty = _ty
        self._oil = _oil
        self._mkt = _mkt
        self._va = _va
        self._gr = _gr

        self.snp = snp
        self.label = label

    def __repr__(self):
        return f"{self.emp, self.snp, self.label}"

    # Getters and Setters for each attribute
    def get_emp(self):
        return self.emp

    def set_emp(self, emp):
        self.emp = emp

    def get_pe(self):
        return self.pe

    def set_pe(self, pe):
        self.pe = pe

    def get_cape(self):
        return self.cape

    def set_cape(self, cape):
        self.cape = cape

    def get_dy(self):
        return self.dy

    def set_dy(self, dy):
        self.dy = dy

    def get_rho(self):
        return self.rho

    def set_rho(self, rho):
        self.rho = rho

    def get_mov(self):
        return self.mov

    def set_mov(self, mov):
        self.mov = mov

    def get_ir(self):
        return self.ir

    def set_ir(self, ir):
        self.ir = ir

    def get_rr(self):
        return self.rr

    def set_rr(self, rr):
        self.rr = rr

    def get_y02(self):
        return self.y02

    def set_y02(self, y02):
        self.y02 = y02

    def get_y10(self):
        return self.y10

    def set_y10(self, y10):
        self.y10 = y10

    def get_stp(self):
        return self.stp

    def set_stp(self, stp):
        self.stp = stp

    def get_cf(self):
        return self.cf

    def set_cf(self, cf):
        self.cf = cf

    def get_mg(self):
        return self.mg

    def set_mg(self, mg):
        self.mg = mg

    def get_rv(self):
        return self.rv

    def set_rv(self, rv):
        self.rv = rv

    def get_ed(self):
        return self.ed

    def set_ed(self, ed):
        self.ed = ed

    def get_un(self):
        return self.un

    def set_un(self, un):
        self.un = un

    def get_gdp(self):
        return self.gdp

    def set_gdp(self, gdp):
        self.gdp = gdp

    def get_m2(self):
        return self.m2

    def set_m2(self, m2):
        self.m2 = m2

    def get_cpi(self):
        return self.cpi

    def set_cpi(self, cpi):
        self.cpi = cpi

    def get_dil(self):
        return self.dil

    def set_dil(self, dil):
        self.dil = dil

    def get_yss(self):
        return self.yss

    def set_yss(self, yss):
        self.yss = yss

    def get_nyf(self):
        return self.nyf

    def set_nyf(self, nyf):
        self.nyf = nyf

    def get_au(self):
        return self._au

    def set_au(self, _au):
        self._au = _au

    def get_dxy(self):
        return self._dxy

    def set_dxy(self, _dxy):
        self._dxy = _dxy

    def get_lcp(self):
        return self._lcp

    def set_lcp(self, _lcp):
        self._lcp = _lcp

    def get_ty(self):
        return self._ty

    def set_ty(self, _ty):
        self._ty = _ty

    def get_oil(self):
        return self._oil

    def set_oil(self, _oil):
        self._oil = _oil

    def get_mkt(self):
        return self._mkt

    def set_mkt(self, _mkt):
        self._mkt = _mkt

    def get_va(self):
        return self._va

    def set_va(self, _va):
        self._va = _va

    def get_gr(self):
        return self._gr

    def set_gr(self, _gr):
        self._gr = _gr

    def get_snp(self):
        return self.snp
    
    def set_snp(self, snp):
        self.snp = snp

    def get_label(self):
        return self.label
    
    def set_label(self, labels):
        self.label = labels

    def finding_in_obj_list(self, objects, target_date):
        return [obj for obj in objects if obj.date == target_date]
    
    def remove_item(self, attribute, item):
        for symbol in attribute:
            getattr(self, symbol).remove(item)

def read_data(self, path):

    file = openpyxl.load_workbook(path, data_only=True)
    data_sheet = file['US']

    emp = []
    pe = []
    cape = []
    dy = []
    rho = []
    mov = []
    ir = []
    rr = []
    y02 = []
    y10 = []
    stp = []
    cf = []
    mg = []
    rv = []
    ed = []
    un = []
    gdp = []
    m2 = []
    cpi = []
    dil = []
    yss = []
    nyf = []
    _au = []
    _dxy = []
    _lcp = []
    _ty = []
    _oil = []
    _mkt = []
    _va = []
    _gr = []

    for row in data_sheet.iter_rows():
        row_data = [cell.value for cell in row]

        if row_data[0] is not None:
            date_of_info = row_data[0].date()
            #print(date_of_info)
            emp.append(DataPoint(date_of_info, row_data[1]))
            pe.append(DataPoint(date_of_info, row_data[2]))
            cape.append(DataPoint(date_of_info, row_data[3]))
            dy.append(DataPoint(date_of_info, row_data[4]))
            rho.append(DataPoint(date_of_info, row_data[5]))
            mov.append(DataPoint(date_of_info, row_data[6]))
            ir.append(DataPoint(date_of_info, row_data[7]))
            rr.append(DataPoint(date_of_info, row_data[8]))
            y02.append(DataPoint(date_of_info, row_data[9]))
            y10.append(DataPoint(date_of_info, row_data[10]))
            stp.append(DataPoint(date_of_info, row_data[11]))
            cf.append(DataPoint(date_of_info, row_data[12]))
            mg.append(DataPoint(date_of_info, row_data[13]))
            rv.append(DataPoint(date_of_info, row_data[14]))
            ed.append(DataPoint(date_of_info, row_data[15]))
            un.append(DataPoint(date_of_info, row_data[16]))
            gdp.append(DataPoint(date_of_info, row_data[17]))
            m2.append(DataPoint(date_of_info, row_data[18]))
            cpi.append(DataPoint(date_of_info, row_data[19]))
            dil.append(DataPoint(date_of_info, row_data[20]))
            yss.append(DataPoint(date_of_info, row_data[21]))
            nyf.append(DataPoint(date_of_info, row_data[22]))
            _au.append(DataPoint(date_of_info, row_data[23]))
            _dxy.append(DataPoint(date_of_info, row_data[24]))
            _lcp.append(DataPoint(date_of_info, row_data[25]))
            _ty.append(DataPoint(date_of_info, row_data[26]))
            _oil.append(DataPoint(date_of_info, row_data[27]))
            _mkt.append(DataPoint(date_of_info, row_data[28]))
            _va.append(DataPoint(date_of_info, row_data[29]))
            _gr.append(DataPoint(date_of_info, row_data[30]))

    self.set_au(_au)
    self.set_cape(cape)
    self.set_cf(cf)
    self.set_cpi(cpi)
    self.set_dil(dil)
    self.set_dxy(_dxy)
    self.set_dy(dy)
    self.set_ed(ed)
    self.set_emp(emp)
    self.set_gdp(gdp)
    self.set_gr(_gr)
    self.set_ir(ir)
    self.set_lcp(_lcp)
    self.set_m2(m2)
    self.set_mg(mg)
    self.set_mkt(_mkt)
    self.set_mov(mov)
    self.set_nyf(nyf)
    self.set_oil(_oil)
    self.set_pe(pe)
    self.set_rho(rho)
    self.set_rr(rr)
    self.set_rv(rv)
    self.set_stp(stp)
    self.set_ty(_ty)
    self.set_un(un)
    self.set_va(_va)
    self.set_y02(y02)
    self.set_y10(y10)
    self.set_yss(yss)

    return self

#info
# [0] Open
# [1] High
# [2] Low
# [3] Close
# [4] Volume
# [5] Dividends
# [6] Stock Splits

def request_snp(self):
    symbol = "^GSPC"

    datapoints = []

    snp_ticker = yfinance.Ticker(symbol)
    snp_data = snp_ticker.history(period="100y", interval="1d", start="1988-01-10", end="2024-05-05", auto_adjust=True)

    # for item, values in snp_data.items():
    #     print(values)

    for date, value in snp_data.iterrows():
        #
        make_date = f"{date.year}-{date.month}-{date.day}"
        if date.month < 10:
            make_date = f"{date.year}-0{date.month}-{date.day}"
        if date.day < 10:
            make_date = f"{date.year}-{date.month}-0{date.day}"
        if date.month < 10 and date.day < 10:
            make_date = f"{date.year}-0{date.month}-0{date.day}"
        #print(make_date)
        make_value = (value['Open'] + value['High'] + value['Low'] + value['Close'])/4
        make_volume = value['Volume']
        make_dividends = value['Dividends']
        make_stock_splits = value['Stock Splits']
        newobj = DataPointStock(date=make_date, value=make_value, volume=make_volume, dividends=make_dividends, stock_splits=make_stock_splits)
        datapoints.append(newobj)

        #print(value['Open'])

    self.set_snp(datapoints)

# labeling data
def labeling_data(self=Data):
    """
    The idea of the labeling follows the approach that all data is getting included in a big grid. From there we adjust the SnP as a new column next to it. After making sure that all data is together and adjusted
    we put according to a growing or decreasing snp a label of either 0 for decrease or equal and 1 for grwoth in it. 
    """

    dates_emp = [str(obj.date) for obj in self.get_emp()]
    dates_pe = {str(obj.date): obj for obj in self.get_pe()}
    dates_cape = [str(obj.date) for obj in self.get_cape()]
    dates_dy = [str(obj.date) for obj in self.get_dy()]
    dates_rho = [str(obj.date) for obj in self.get_rho()]
    dates_mov = [str(obj.date) for obj in self.get_mov()]
    dates_ir = [str(obj.date) for obj in self.get_ir()]
    dates_rr = [str(obj.date) for obj in self.get_rr()]
    dates_y02 = [str(obj.date) for obj in self.get_y02()]
    dates_y10 = [str(obj.date) for obj in self.get_y10()]
    dates_stp = [str(obj.date) for obj in self.get_stp()]
    dates_cf = [str(obj.date) for obj in self.get_cf()]
    dates_mg = [str(obj.date) for obj in self.get_mg()]
    dates_rv = [str(obj.date) for obj in self.get_rv()]
    dates_ed = [str(obj.date) for obj in self.get_ed()]
    dates_un = [str(obj.date) for obj in self.get_un()]
    dates_gdp = [str(obj.date) for obj in self.get_gdp()]
    dates_m2 = [str(obj.date) for obj in self.get_m2()]
    dates_cpi = [str(obj.date) for obj in self.get_cpi()]
    dates_dil = [str(obj.date) for obj in self.get_dil()]
    dates_yss = [str(obj.date) for obj in self.get_yss()]
    dates_nyf = [str(obj.date) for obj in self.get_nyf()]
    dates__au = [str(obj.date) for obj in self.get_au()]
    dates__dxy = [str(obj.date) for obj in self.get_dxy()]
    dates__lcp = [str(obj.date) for obj in self.get_lcp()]
    dates__ty = [str(obj.date) for obj in self.get_ty()]
    dates__oil = [str(obj.date) for obj in self.get_oil()]
    dates__mkt = [str(obj.date) for obj in self.get_mkt()]
    dates__va = [str(obj.date) for obj in self.get_va()]
    dates__gr = [str(obj.date) for obj in self.get_gr()]

    valid_snp_data = []

    new_list_emp = []
    new_list_pe = []
    new_list_cape = []
    new_list_dy = []
    new_list_rho = []
    new_list_mov = []
    new_list_ir = []
    new_list_rr = []
    new_list_y02 = []
    new_list_y10 = []
    new_list_stp = []
    new_list_cf = []
    new_list_mg = []
    new_list_rv = []
    new_list_ed = []
    new_list_un = []
    new_list_gdp = []
    new_list_m2 = []
    new_list_cpi = []
    new_list_dil = []
    new_list_yss = []
    new_list_nyf = []
    new_list__au = []
    new_list__dxy = []
    new_list__lcp = []
    new_list__ty = []
    new_list__oil = []
    new_list__mkt = []
    new_list__va = []
    new_list__gr = []
    new_list_snp = []

    for item in self.get_snp():

        some_date = str(item.date)
        date_split = some_date.split("-")
        date_string_low = date_split[0] + "-" + date_split[1] + "-" + str(int(date_split[2]) - 1)
        date_string_mid = date_split[0] + "-" + date_split[1] + "-" + str(int(date_split[2]))
        date_string_high = date_split[0] + "-" + date_split[1] + "-" + str(int(date_split[2]) + 1)

        if len(date_string_low) < 10:
            checking_low = date_string_low.split("-")
            if int(checking_low[1]) < 10:
                checking_low[1] = "0" + str(int(checking_low[1]))
            if int(checking_low[2]) < 10:
                checking_low[2] = "0" + str(int(checking_low[1]))
            checking_low = checking_low[0] + "-" + checking_low[1] + "-" + checking_low[2]
            date_string_low = checking_low

        if date_string_low in dates_emp:
            #print(date_string_low, date_string_mid)
            valid_snp_data.append(item)
    
    self.snp = valid_snp_data


    # for i, stock in zip(self.get_emp(), self.get_snp()):
    #     print(f"emp: {i} dividends: {stock.get_dividends()} stock splits: {stock.get_stock_splits()}")


    first_obj = next(iter(self.get_snp()))
    first_str_split = first_obj.date.split("-")
    first_obj_year = int(first_str_split[0])
    first_obj_month = int(first_str_split[1])
    first_obj_day = int(first_str_split[2])

    last_obj = next(reversed(self.get_snp()))
    #last_obj_date = last_obj.get_date()
    last_str_split = last_obj.date.split("-")
    last_obj_year = int(last_str_split[0])
    last_obj_month = int(last_str_split[1])
    last_obj_day = int(last_str_split[2])

    while len(self.get_snp()) != len(self.get_emp()):
        snd_iteration_valid_data = []
        #print(len(self.get_snp()), len(self.get_emp()))
        close_date_snp_data = {obj.date:obj for obj in self.get_snp()}
        #print(close_date_snp_data.keys())
        counter = 0
        for item in self.get_emp():
            #print(counter)
            counter += 1
            initial_date = str(item.date)
            date_down = initial_date
            found_down = False
            date_up = initial_date
            found_up = False
            
            #print(initial_date)

            if item.date in close_date_snp_data.keys():
                snd_iteration_valid_data.append(close_date_snp_data[item.date])
                #self.snp.append(close_date_snp_data[item.date])
                break
            else:
                current_date = date_down
                while found_down == False:

                    current_date = current_date.split("-")
                    current_year = int(current_date[0])
                    current_month = int(current_date[1])
                    current_day = int(current_date[2])

                    if current_year <= first_obj_year and current_month <= first_obj_month and current_day <= first_obj_day: # wenn alle zahlen kleiner als die des ersten entry, dann 
                        final_downer_date = next(iter(close_date_snp_data))
                        found_down == True
                        value_down = close_date_snp_data[final_downer_date].get_value()
                        volume_down = close_date_snp_data[final_downer_date].get_volume()
                        dividends_down = close_date_snp_data[final_downer_date].get_dividends()
                        stock_splits_up = close_date_snp_data[final_downer_date].get_stock_splits()
                        break

                    #print(current_year, current_month, current_day)

                    if current_month >= 1 and current_day > 1: # We want to make the date -1 day in case there are still dates of the month left.
                        current_day -= 1
                    elif current_month > 1 and current_day == 1: # we want to make the month to -1 in case 
                        current_month -= 1
                        current_day = 31
                    elif current_month == 1 and current_day == 1:
                        current_year -= 1
                        current_month = 12
                        current_day = 31
                    if len(str(current_day)) < 2:
                        current_day = "0" + str(current_day)
                    if len(str(current_month)) < 2:
                        current_month = "0" + str(current_month)

                    # Datum zusammensetzen:
                    current_date = str(current_year) +"-"+ str(current_month) +"-"+ str(current_day)

                    if current_date in close_date_snp_data.keys():
                        found_down = True
                        value_down = close_date_snp_data[current_date].get_value()
                        volume_down = close_date_snp_data[current_date].get_volume()
                        dividends_down = close_date_snp_data[current_date].get_dividends()
                        stock_splits_down = close_date_snp_data[current_date].get_stock_splits()

                #print("Found the lower")

                current_date = date_up
                while found_up == False:
                    
                    current_date = current_date.split("-")
                    current_year = int(current_date[0])
                    current_month = int(current_date[1])
                    current_day = int(current_date[2])

                    if current_year >= last_obj_year and current_month >= last_obj_month and current_day >= last_obj_day:
                        #print("so yes we assigned the top edge")
                        final_upper_date = next(iter(reversed(close_date_snp_data)))
                        found_up == True
                        value_up = close_date_snp_data[final_upper_date].get_value()
                        volume_up = close_date_snp_data[final_upper_date].get_volume()
                        dividends_up = close_date_snp_data[final_upper_date].get_dividends()
                        stock_splits_up = close_date_snp_data[final_upper_date].get_stock_splits()
                        break

                    if current_month <= 12 and current_day < 31: 
                        current_day += 1
                    elif current_month < 12 and current_day == 31:
                        current_month += 1
                        current_day = 1
                    elif current_month == 12 and current_day == 31:
                        current_year += 1
                        current_month = 1
                        current_day = 1
                    if len(str(current_day)) < 2:
                        current_day = "0" + str(current_day)
                    if len(str(current_month)) < 2:
                        current_month = "0" + str(current_month)

                    # Datum zusammensetzen:
                    current_date = str(current_year) +"-"+ str(current_month) +"-"+ str(current_day)

                    if current_date in close_date_snp_data.keys():
                        found_up = True
                        value_up = close_date_snp_data[current_date].get_value()
                        volume_up = close_date_snp_data[current_date].get_volume()
                        dividends_up = close_date_snp_data[current_date].get_dividends()
                        stock_splits_up = close_date_snp_data[current_date].get_stock_splits()

                difference_value = (value_up - value_down) / 2
                if difference_value < 0:
                    difference_value = difference_value * -1

                difference_volume = (volume_up - volume_down) / 2
                if difference_volume < 0:
                    difference_volume = difference_volume * -1

                response_value = value_down + difference_value
                response_volume = volume_down + difference_volume
                response_dividends = 0
                response_stock_splits = 0

                #print(item.date)
                provisional_datapoint_stock = DataPointStock(item.date, response_value, response_volume)

                #self.snp.append(provisional_datapoint_stock)
                snd_iteration_valid_data.append(provisional_datapoint_stock)

                self.set_snp(snd_iteration_valid_data)

    label = []
    snp_data_for_labeling = self.get_snp()

    prior_item = snp_data_for_labeling[0]
    label.append(1)  
    for item in snp_data_for_labeling[1:]:
        if item.get_value() > prior_item.value:
            label.append(1)
        else:
            label.append(0)
        prior_item = item
    
    self.set_label(label)
    #print(self.get_label())
    #print(len(self.get_emp()), len(self.get_snp()), len(self.get_label()))

    # for item, otheritem, anotheritem in zip(self.get_emp(), self.get_snp(), self.get_label()):
    #     print(f"Datum emp: {item.get_date()} datum snp: {otheritem.get_date()}\t value snp: {otheritem.get_value()}\t volume: {otheritem.get_volume()}\t dividends: {otheritem.get_dividends()}\t stock_splits:{otheritem.get_stock_splits()}\t  label: {anotheritem}")
        

    snp_data = self.get_snp()
    #print(len(snp_data), len(self.get_emp()), len(self.get_au()))
    if not snp_data:
        raise ValueError("No data returned by get_snp()")

    # Extract data from DataPointStock objects
    data = [
        {
            'date': dp.get_date(),
            'value': dp.get_value(),
            'volume': dp.get_volume(),
            'dividends': dp.get_dividends(),
            'stock_splits': dp.get_stock_splits()
        }
        for dp in snp_data
    ]

def create_csv(self, filenamettrain='training_data.csv', filenametest='validation_data.csv'):
        # Anzahl der Datenpunkte und Features
        n_samples = len(self.emp)
        n_features = 31  # 30 features plus SNP

        # Initialisiere Arrays für Features (X) und Labels (y)
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)

        data_list = []
        for emp, pe, cape, dy, rho, mov, ir, rr, y02, y10, stp, cf, mg, rv, ed, un, gdp, m2, cpi, dil, yss, nyf, _au, _dxy, _lcp, _ty, _oil, _mkt, _va, _gr, snp, label in zip(self.emp, self.pe, self.cape, self.dy, self.rho, self.mov, self.ir, self.rr, self.y02, self.y10, self.stp, self.cf, self.mg, self.rv, self.ed, self.un, self.gdp, self.m2, self.cpi, self.dil, self.yss, self.nyf, self._au, self._dxy, self._lcp, self._ty, self._oil, self._mkt, self._va, self._gr, self.snp, self.label):
            one_datapoint = FullDataPoint(emp=emp.get_value(), pe=pe.get_value(), cape=cape.get_value(), dy=dy.get_value(), rho=rho.get_value(), mov=mov.get_value(), ir=ir.get_value(), rr=rr.get_value(), y02=y02.get_value(), y10=y10.get_value(), stp=stp.get_value(), cf=cf.get_value(), mg=mg.get_value(), rv=rv.get_value(), ed=ed.get_value(), un=un.get_value(), gdp=gdp.get_value(), m2=m2.get_value(), cpi=cpi.get_value(), dil=dil.get_value(), yss=yss.get_value(), nyf=nyf.get_value(), _au=_au.get_value(), _dxy=_dxy.get_value(), _lcp=_lcp.get_value(), _ty=_ty.get_value(), _oil=_oil.get_value(), _mkt=_mkt.get_value(), _va=_va.get_value(), _gr=_gr.get_value(),snp=snp.get_value(), label=label)
            data_list.append(one_datapoint)

        for i, data_point in enumerate(data_list):
            X[i, 0] = data_point.emp
            X[i, 1] = data_point.pe
            X[i, 2] = data_point.cape
            X[i, 3] = data_point.dy
            X[i, 4] = data_point.rho
            X[i, 5] = data_point.mov
            X[i, 6] = data_point.ir
            X[i, 7] = data_point.rr
            X[i, 8] = data_point.y02
            X[i, 9] = data_point.y10
            X[i, 10] = data_point.stp
            X[i, 11] = data_point.cf
            X[i, 12] = data_point.mg
            X[i, 13] = data_point.rv
            X[i, 14] = data_point.ed
            X[i, 15] = data_point.un
            X[i, 16] = data_point.gdp
            X[i, 17] = data_point.m2
            X[i, 18] = data_point.cpi
            X[i, 19] = data_point.dil
            X[i, 20] = data_point.yss
            X[i, 21] = data_point.nyf
            X[i, 22] = data_point._au
            X[i, 23] = data_point._dxy
            X[i, 24] = data_point._lcp
            X[i, 25] = data_point._ty
            X[i, 26] = data_point._oil
            X[i, 27] = data_point._mkt
            X[i, 28] = data_point._va
            X[i, 29] = data_point._gr
            X[i, 30] = data_point.snp
            y[i] = data_point.label

        # Split data into training and testing sets
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Define the header
        header = ['emp', 'pe', 'cape', 'dy', 'rho', 'mov', 'ir', 'rr', 'y02', 'y10', 'stp', 'cf', 'mg', 'rv', 'ed', 'un', 'gdp', 'm2', 'cpi', 'dil', 'yss', 'nyf', '_au', '_dxy', '_lcp', '_ty', '_oil', '_mkt', '_va', '_gr', 'snp', 'label']

        # Write the data to CSV
        with open(filenamettrain, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for i in range(len(X_train)):
                writer.writerow(X_train[i].tolist() + [y_train[i]])

        # Write the data to CSV
        with open(filenametest, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for i in range(len(X_test)):
                writer.writerow(X_test[i].tolist() + [y_test[i]])

def create_svg(self, filename='output.svg'):
    # Anzahl der Datenpunkte und Features
    n_samples = len(self.emp)
    n_features = 31  # 30 features plus SNP

    # Initialisiere Arrays für Features (X) und Labels (y)
    X = np.zeros((n_samples, n_features + 1))
    y = np.zeros(n_samples)

    data_list = []
    for emp, pe, cape, dy, rho, mov, ir, rr, y02, y10, stp, cf, mg, rv, ed, un, gdp, m2, cpi, dil, yss, nyf, _au, _dxy, _lcp, _ty, _oil, _mkt, _va, _gr, snp, label in zip(self.emp, self.pe, self.cape, self.dy, self.rho, self.mov, self.ir, self.rr, self.y02, self.y10, self.stp, self.cf, self.mg, self.rv, self.ed, self.un, self.gdp, self.m2, self.cpi, self.dil, self.yss, self.nyf, self._au, self._dxy, self._lcp, self._ty, self._oil, self._mkt, self._va, self._gr, self.snp, self.label):
        one_datapoint = FullDataPoint(emp=emp.get_value(), pe=pe.get_value(), cape=cape.get_value(), dy=dy.get_value(), rho=rho.get_value(), mov=mov.get_value(), ir=ir.get_value(), rr=rr.get_value(), y02=y02.get_value(), y10=y10.get_value(), stp=stp.get_value(), cf=cf.get_value(), mg=mg.get_value(), rv=rv.get_value(), ed=ed.get_value(), un=un.get_value(), gdp=gdp.get_value(), m2=m2.get_value(), cpi=cpi.get_value(), dil=dil.get_value(), yss=yss.get_value(), nyf=nyf.get_value(), _au=_au.get_value(), _dxy=_dxy.get_value(), _lcp=_lcp.get_value(), _ty=_ty.get_value(), _oil=_oil.get_value(), _mkt=_mkt.get_value(), _va=_va.get_value(), _gr=_gr.get_value(),snp=snp.get_value(), label=label)
        data_list.append(one_datapoint)

    for i, data_point in enumerate(data_list):
        X[i, 0] = data_point.emp
        X[i, 1] = data_point.pe
        X[i, 2] = data_point.cape
        X[i, 3] = data_point.dy
        X[i, 4] = data_point.rho
        X[i, 5] = data_point.mov
        X[i, 6] = data_point.ir
        X[i, 7] = data_point.rr
        X[i, 8] = data_point.y02
        X[i, 9] = data_point.y10
        X[i, 10] = data_point.stp
        X[i, 11] = data_point.cf
        X[i, 12] = data_point.mg
        X[i, 13] = data_point.rv
        X[i, 14] = data_point.ed
        X[i, 15] = data_point.un
        X[i, 16] = data_point.gdp
        X[i, 17] = data_point.m2
        X[i, 18] = data_point.cpi
        X[i, 19] = data_point.dil
        X[i, 20] = data_point.yss
        X[i, 21] = data_point.nyf
        X[i, 22] = data_point._au
        X[i, 23] = data_point._dxy
        X[i, 24] = data_point._lcp
        X[i, 25] = data_point._ty
        X[i, 26] = data_point._oil
        X[i, 27] = data_point._mkt
        X[i, 28] = data_point._va
        X[i, 29] = data_point._gr
        X[i, 30] = data_point.snp
        y[i] = data_point.label

    # Create SVG file
    svg_header = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="800" height="800">'
    svg_footer = '</svg>'
    svg_content = ''

    # Let's assume we want to plot 'emp' vs 'pe'
    max_emp = np.max(X[:, 0])
    max_pe = np.max(X[:, 1])

    # Define scaling factors for better visualization
    scale_emp = 700 / max_emp
    scale_pe = 700 / max_pe

    for i in range(n_samples):
        emp_scaled = X[i, 0] * scale_emp
        pe_scaled = X[i, 1] * scale_pe
        svg_content += f'<circle cx="{emp_scaled + 50}" cy="{750 - pe_scaled}" r="4" fill="blue" />\n'

    with open(filename, 'w') as f:
        f.write(svg_header + "\n")
        f.write(svg_content)
        f.write(svg_footer)
    