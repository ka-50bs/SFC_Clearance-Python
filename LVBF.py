# -*- coding: utf-8 -*-
"""
LVBF
====

This module provides tools for reading and writing LabVIEW binary files. For
the time being only two types of files are considered, flattened data and TDMS.

Two classes are defined, one for each type of file:
    LV_fd works with flattened data
    LV_TDMS works with Technical Data Management Streaming (TDMS)

The former is more advanced in terms of development, although not all types of
flattened data are implemented, the latter is still in an experimental stage.

@author: pxcandeias
"""

# Developed for Python 3.6.5

# Python standard library
from time import time
import os.path
import platform

# 3rd party modules
import numpy as np


class LV_fd():
    """LabVIEW flattened data.
    
    Instance variables
    ------------------
    fobj # file object for the read/write functions
    endian # file endianness
    encoding # byte encoding (Python 3)
    
    Methods
    -------
    _set_dtypes # set data attributes
    read_numeric
    read_boolean
    read_string
    read_path
    read_array
    read_timestamp
    read_waveform
    read_cluster
    write_numeric
    write_boolean
    write_string
    write_path
    write_array
    write_timestamp
    write_waveform
    write_cluster
    Unix_time
    LV_time
    EOD # End-Of-Data
    
    Data attributes
    ---------------
    self.LVint8 # 8-bit signed integer
    self.LVuint8 # 8-bit unsigned integer
    self.LVint16 # 16-bit signed integer
    self.LVuint16 # 16-bit unsigned integer
    self.LVint32 # 32-bit signed integer
    self.LVuint32 # 32-bit unsigned integer
    self.LVint64 # 64-bit signed integer
    self.LVuint64 # 64-bit unsigned integer
    self.LVfloat16 # 16-bit floating-point number
    self.LVfloat32 # 32-bit floating-point number
    self.LVfloat64 # 64-bit floating-point number
    self.LVstring1 # 1-character (8-bit) string
    self.LVtimestamp # time stamp
    
    References
    ----------
    http://zone.ni.com/reference/en-XX/help/371361P-01/lvconcepts/flattened_data/
    """
    def __init__(self, endian='>', encoding='cp1252'):
        """Set initial data.
        
        Parameters
        ----------
        endian : str, optional
            File endianness (MSB by default, see link in `LV_fd()`).
            Default is '>' (big-endian).
        encoding : str, optional
            Byte encoding ('cp1252' for Windows, 'utf-8' for Linux).
            Default is 'cp1252'.
        """
        self.fobj = None # file object for the read/write functions
        self.endian = endian # file endianness
        self.encoding = encoding # byte encoding (Python 3)
        self._set_dtypes()
        
    def _set_dtypes(self):
        """Set numpy data types consistent with LabVIEW data types.
        
        References
        ----------
        http://zone.ni.com/reference/en-XX/help/371361J-01/lvhowto/numeric_data_types_table/
        https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
        """
        self.LVint8 = np.dtype(f'{self.endian}i1') # 8-bit signed integer
        self.LVuint8 = np.dtype(f'{self.endian}u1') # 8-bit unsigned integer
        self.LVint16 = np.dtype(f'{self.endian}i2') # 16-bit signed integer
        self.LVuint16 = np.dtype(f'{self.endian}u2') # 16-bit unsigned integer
        self.LVint32 = np.dtype(f'{self.endian}i4') # 32-bit signed integer
        self.LVuint32 = np.dtype(f'{self.endian}u4') # 32-bit unsigned integer
        self.LVint64 = np.dtype(f'{self.endian}i8') # 64-bit signed integer
        self.LVuint64 = np.dtype(f'{self.endian}u8') # 64-bit unsigned integer
        self.LVfloat16 = np.dtype(f'{self.endian}f2') # 16-bit floating-point number
        self.LVfloat32 = np.dtype(f'{self.endian}f4') # 32-bit floating-point number
        self.LVfloat64 = np.dtype(f'{self.endian}f8') # 64-bit floating-point number
        self.LVstring1 = np.dtype(f'{self.endian}S1') # 1-character (8-bit) string
        self.LVtimestamp = np.dtype([('seconds', self.LVint64),
                                     ('fraction', self.LVuint64)])

    def read_numeric(self, d, c=1):
        """Read LabVIEW numeric data type.
        
        Parameters
        ----------
        d : numpy.dtype
            Number data type.
        c : int, optional
            Data count.
            Default is 1.
        
        Returns
        -------
        val : array_like
            data read
        """
        val = np.fromfile(self.fobj, dtype=d, count=c, sep='')
        
        return np.asarray(val[0], dtype=d) if c==1 else val

    def read_boolean(self, d=None, c=1):
        """Read LabVIEW boolean data type.
        
        Parameters
        ----------
        d : None, optional
            Boolean data type. This parameter is actually ignored, only kept to
            have the same number of arguments on all basic read_* functions.
            Internally `self.LVint8` is used.
        c : int, optional
            Data count.
            Default is 1.
        
        Returns
        -------
        val : array_like
            Boolean value(s).
        """
        return self.read_numeric(self.LVint8, c)

    def read_string(self, d=None, c=1):
        """Read LabVIEW string data type.
        
        Parameters
        ----------
        d : None, optional
            String data type. This parameter is actually ignored, only kept to
            have the same number of arguments on all basic read_* functions.
            Internally `self.LVstring1` is used.
        c : int, optional
            Data count.
            Default is 1.
       
        Returns
        -------
        val : str
            String read.
        
        Notes
        -----
        The string is internally decoded from bytes (Python 3) using the
        `self.encoding` data attribute.
        
        Examples
        --------
        >>> import io
        >>> f = io.BytesIO(bytes.fromhex('00000003414243'))
        
        """
        val = []
        
        for v in range(c):
            temp = self.read_array(self.read_numeric, self.LVstring1)
            val.append(temp.tobytes().decode(self.encoding)) # bytes->string
            
        return val

    def read_path(self, d=None, c=1):
        """Read LabVIEW path data type.
        
        Parameters
        ----------
        d : None, optional
            Path data type. This parameter is actually ignored, only kept to
            have the same number of arguments on all basic read_* functions.
            Internally `self.LVstring1` is used.
        c : int, optional
            Data count.
            Default is 1.
        
        Returns
        -------
        val : str
            File path.
        
        Warnings
        --------
        This function has not been tested yet.
        
        See also
        --------
        `self.read_string()`
        
        Examples
        --------
        >>> import io
        >>> b = io.BytesIO(bytes.fromhex('505448300000000B0000000201430466696C65'))
        >>> a = LV_fd()
        >>> a.fobj = io.BufferedRandom(b)
        >>> a.read_path()
        
        """
        val = []
        
        for v in range(c):
            temp = self.read_numeric(self.LVstring1, 4)
            
            if temp.tobytes().decode(self.encoding) == 'PTH0':
                nbytes = self.read_numeric(self.LVint32) # number of bytes
                ncomps = self.read_numeric(self.LVint32) # number of components
                nbytes -= self.LVint32.itemsize
                
                for v in range(ncomps):
                    nchars = self.read_numeric(self.LVuint8)
                    nbytes -= self.LVuint8.itemsize
                    chars = self.read_numeric(self.LVstring1, nchars)
                    nbytes -= nchars*self.LVstring1.itemsize
                    
                    if (v == 0) and (platform.system() == 'Windows'):
                        chars = np.append(chars, np.array(':'))
                    
                    val.append(chars.tobytes().decode(self.encoding))
                
                if nbytes != 0:
                    raise ValueError(f'Wrong number of bytes ({nbytes}).')
                
            else:
                raise ValueError(f'Wrong path string ({temp}).')
            
        return os.path.join(*val)

    def read_array(self, f, d=None, ndims=1):
        """Read LabVIEW array data type.
        
        Parameters
        ----------
        f : function
            Array read function.
        d : numpy.dtype or None, optional
            Array data type.
        ndims : int, optional
            Number of array dimensions.
            Default is 1.
        
        Returns
        -------
        val : ndarray or list
            Output array.
        
        Notes
        -----
        The type of `val` depends on the type of data returned by `f`.
        
        Examples
        --------
        >>> import io
        >>> b = io.BytesIO(bytes.fromhex('0000 0002 0000 0003 0102 0304 0506'))
        >>> a = LV_fd()
        >>> a.fobj = io.BufferedRandom(b)
        >>> a.read_array()
        
        """
        shape = self.read_numeric(self.LVint32, ndims)
        val = f(d, np.prod(shape))
        
        if isinstance(val, np.ndarray):
            val = np.asarray(val, dtype=d).reshape(shape)
        elif ndims == 2:
            val = [val[shape[1]*v:shape[1]*(v+1)] for v in range(shape[0])]
            
        return val

    def read_timestamp(self, d=None, c=1):
        """Read LabVIEW time stamp data type.
        
        Parameters
        ----------
        d : None, optional
            Data type. This parameter is actually ignored, only used to keep
            the same number of arguments on all read_* functions.
            Internally `self.LVtimestamp` is used.
        c : int, optional
            Data count.
            Default is 1.
        
        Returns
        -------
        val : timestamp
            Time stamp.
        
        References
        ----------
        http://www.ni.com/white-paper/7900/en
        """
        val = self.read_numeric(self.LVtimestamp, c)
        
        return val

    def read_waveform(self, d):
        """Read LabVIEW waveform data type.
        
        Parameters
        ----------
        d : numpy.dtype
            Array data type.
        
        Returns
        -------
        t0, dt, y : tuple
            LabVIEW waveform.
        
        References
        ----------
        http://digital.ni.com/public.nsf/allkb/B965F316364DE17B862572DF00363B10
        """
        t0 = self.read_timestamp()
        dt = self.read_numeric(self.LVfloat64)
        y = self.read_array(self.read_numeric, d)
        
        return (t0, dt, y)

    def read_cluster(self, f, d):
        """Read LabVIEW cluster data type.
        
        Parameters
        ----------
        f : iterable
            Cluster read function(s).
        d : iterable
            Cluster type(s).
        
        Returns
        -------
        val : list
            Cluster value(s).
        """
        val = []
        
        for read_, dtype in zip(f, d):
            
            if isinstance(read_, tuple): # read_array or read_cluster
                val.append(read_[0](read_[1], dtype))
            else:
                val.append(read_(dtype))
            
        return val

    def write_numeric(self, val):
        """Write LabVIEW numeric data type.
        
        Parameters
        ----------
        val : ndarray
            Numeric value(s).
        
        Returns
        -------
        None
        """
        val.tofile(self.fobj, sep='', format='%s')

    def write_boolean(self, val):
        """Write LabVIEW boolean data type.
        
        Parameters
        ----------
        val : boolean
            Boolean value(s).
        
        Returns
        -------
        None
        """
        self.write_numeric(val)

    def write_string(self, val):
        """Write LabVIEW string data type.
        
        Parameters
        ----------
        val : str
            String to be written.
        
        Returns
        -------
        None
        
        Notes
        -----
        The string is internally encoded to bytes (Python 3) using the
        `self.encoding` data attribute.
        """
        for v in val:
            temp = np.fromstring(v.encode(encoding=self.encoding),
                                 dtype=self.LVstring1)
            self.write_array(self.write_numeric, temp)

    def write_path(self, val):
        """Write LabVIEW path data type.
        
        Parameters
        ----------
        val : str
            File path.
        
        Returns
        -------
        None
        
        Warnings
        --------
        This function has not been tested yet.
        
        See also
        --------
        `self.write_string()`
        """
        temp = np.fromstring('PTH0'.encode(encoding=self.encoding),
                             dtype=self.LVstring1)
        self.write_numeric(temp)
        nbytes = len(val)*self.LVstring1.itemsize
        comps = os.path.split(val)
        nbytes += len(comps)*self.LVuint8.itemsize
        self.write_numeric(nbytes)
        
        for chars in comps:
            self.write_numeric(len(chars))
            temp = np.fromstring(chars.encode(encoding=self.encoding),
                                 dtype=self.LVstring1)
            self.write_numeric(temp)

    def write_array(self, f, val):
        """Write LabVIEW array data type.
        
        Parameters
        ----------
        f : function
            Array write function.
        val : ndarray or list
            Input array.
        
        Returns
        -------
        None
        
        Notes
        -----
        The iterable `val` is internally converted to a ndarray prior to
        writing.
        """
        val = np.asarray(val)
        self.write_numeric(np.array(val.shape, dtype=self.LVint32))
        f(val.reshape(-1))

    def write_timestamp(self, val):
        """Write LabVIEW time stamp data type.
        
        Parameters
        ----------
        val : timestamp
            Time stamp.
        
        Returns
        -------
        None
        
        References
        ----------
        http://www.ni.com/white-paper/7900/en
        """
        self.write_numeric(val)

    def write_waveform(self, val):
        """Write LabVIEW waveform data type.
        
        Parameters
        ----------
        val : tuple (t0, dt, y)
            LabVIEW waveform.
        
        Returns
        -------
        None
        
        References
        ----------
        http://digital.ni.com/public.nsf/allkb/B965F316364DE17B862572DF00363B10
        """
        (t0, dt, y) = val
        self.write_timestamp(t0)
        self.write_numeric(dt)
        self.write_array(self.write_numeric, y)

    def write_cluster(self, f, val):
        """Write LabVIEW cluster data type.
        
        Parameters
        ----------
        f : iterable
            Cluster write function(s).
        val : iterable
            Cluster value(s).
        
        Returns
        -------
        None
        """
        for write_, value in zip(f, val):
            
            if isinstance(write_, tuple): # write_array or write_cluster
                write_[0](write_[1], value)
            else:
                write_(value)

    def Unix_time(self, timestamp):
        """LabVIEW timestamp to Unix time conversion.
        
        Parameters
        ----------
        timestamp : ndarray
            LabVIEW timestamp (1904-01-01 00:00:00 UTC)
        
        Returns
        -------
        value : int
            Unix timestamp (1970-01-01 00:00:00 UTC)
        
        Warnings
        --------
        There is a caveat related with the limits of Unix epoch.
        
        References
        ----------
        http://forums.ni.com/t5/LabVIEW/does-labview-provide-epoch-time-converter-support/td-p/452093
        """
        offset = 2082844800 # time offset between LabVIEW epoch and Unix epoch
        tick = 2**-64 # clock tick
        
        if timestamp['seconds'] < offset: # CAVEAT: limits of Unix epoch
            value = timestamp['seconds']+timestamp['fraction']*tick
        else:
            value = timestamp['seconds']-offset+timestamp['fraction']*tick
        
        return value

    def LV_time(self, timestamp):
        """Unix time to LabVIEW timestamp conversion.
        
        Parameters
        ----------
        timestamp : int
            Unix timestamp (1970-01-01 00:00:00 UTC)
        
        Returns
        -------
        out : ndarray
            LabVIEW timestamp (1904-01-01 00:00:00 UTC)
        
        Warnings
        --------
        There is a caveat related with the limits of Unix epoch.
        
        References
        ----------
        http://forums.ni.com/t5/LabVIEW/does-labview-provide-epoch-time-converter-support/td-p/452093
        """
        offset = 2082844800 # time offset between LabVIEW epoch and Unix epoch
        tick = 2**-64 # clock tick
        seconds = int(timestamp)
        
        if seconds+offset < int(time()): # CAVEAT: limits of Unix epoch
            seconds += offset
        
        fraction = (timestamp-int(timestamp))/tick
        
        return np.array((seconds,fraction), dtype=self.LVtimestamp)

    def EOD(self):
        """End-Of-Data.
        
        Paramaters
        ----------
        None
        
        Returns
        -------
        out : bool
            True if end of file, False otherwise.
        """
        return self.fobj.read() == b''


class LV_TDMS(LV_fd):
    """LabVIEW Technical Data Management Streaming (TDMS).
    
    Data attributes
    ---------------
    _filename : string
    endian : file endianness (ToC is always LSB, check ToC_mask for numeric values)
    encoding : byte encoding ('utf-8' by default)
    
    Properties
    ----------
    filename
    eof
    
    Methods
    -------
    read : read TDMS file
    write : write TDMS file
    modify : modify TDMS data in memory
    
    Warnings
    --------
    This class is currently a work in progress.
    
    References
    ----------
    http://www.ni.com/white-paper/5696/en/
    
    http://www.ni.com/tutorial/9334/en/
    http://www.ni.com/white-paper/3727/en/
    http://www.ni.com/white-paper/5696/en/
    http://www.ni.com/white-paper/14252/en/
    http://zone.ni.com/reference/en-XX/help/371361M-01/lvhowto/ni_test_data_exchange/
    http://zone.ni.com/reference/en-XX/help/371361M-01/lvconcepts/fileio_tdms_model/
    http://zone.ni.com/reference/en-XX/help/371361M-01/lvconcepts/fileio_tdms_tdm/
    http://zone.ni.com/reference/en-XX/help/371361M-01/lvconcepts/fileio_tdm_interacting/
    """
    def __init__(self, verbose=True):
        """Set initial data."""
        super().__init__(endian='<', encoding='utf-8') # default for TDMS
        # data attributes
        self._filename = None
        self._segments = []
        self.Paths = {}
        self.Props = []
        
        self.kTocMetaData = 1 << 1
        self.kTocNewObjList = 1 << 2
        self.kTocRawData = 1 << 3
        self.kTocInterleavedData = 1 << 5
        self.kTocBigEndian = 1 << 6
        self.kTocDAQmxRawData = 1 << 7
        
        self.tdsTypeVoid = 0x00 # C enum starts at zero
        self.tdsTypeI8 = 0x01
        self.tdsTypeI16 = 0x02
        self.tdsTypeI32 = 0x03
        self.tdsTypeI64 = 0x04
        self.tdsTypeU8 = 0x05
        self.tdsTypeU16 = 0x06
        self.tdsTypeU32 = 0x07
        self.tdsTypeU64 = 0x08
        self.tdsTypeSingleFloat = 0x09
        self.tdsTypeDoubleFloat = 0x10
        self.tdsTypeExtendedFloat = 0x11
        self.tdsTypeSingleFloatWithUnit = 0x19 # C enum explicit value
        self.tdsTypeDoubleFloatWithUnit = 0x1a
        self.tdsTypeExtendedFloatWithUnit = 0x1b
        self.tdsTypeString = 0x20 # C enum explicit value
        self.tdsTypeBoolean = 0x21
        self.tdsTypeTimeStamp = 0x44 # C enum explicit value
        self.tdsTypeFixedPoint = 0x4F # C enum explicit value
        self.tdsTypeComplexSingleFloat = 0x08000c # C enum explicit value
        self.tdsTypeComplexDoubleFloat = 0x10000d # C enum explicit value
        self.tdsTypeDAQmxRawData = 0xFFFFFFFF # C enum explicit value
# NOTE:
# LabVIEW floating-point types with unit translate into a floating-point
# channel with a property named unit_string that contains the unit as a string.
    
    filename = property(lambda self: self._filename)
    eof = property(lambda self: self.EOD())
    
    def read(self, filename):
        """...
        
        TDMS segment layout:
            Lead In
            Meta Data
            Raw Data
        
        References
        ----------
        http://www.ni.com/white-paper/5696/en/
        """
        self._filename = filename
        
        with open(filename, 'rb') as self.fobj:
            
            while self.read_LeadIn():
                print('\tLeadIn')
                self.check_ToC_mask()
                
                if self.ToC_mask & self.kTocBigEndian:
                    self.endian = '>'
                    self._set_dtypes()
                
                if self.ToC_mask & self.kTocMetaData:
                    self.read_MetaData()
#                if self.ToC_mask & self.kTocRawData:
#                    if not (self.RDI & 0xFFFFFFFF):
#                        self.read_RawData()
                
                self.fobj.seek(int(self.RemSegLen)+self._segments[-1][0], 0)
        
        return self
    
    def read_LeadIn(self):
        """...
        
        Lead In layout:
            TDSm tag (4-byte tag that identifies a TDMS segment, "TDSm")
            ToC (four bytes used as bit mask, see kToc* in __init__)
            Version number (32-bit unsigned integer)
                4712 corresponds to the TDMS file format version 1.0
                4713 corresponds to the TDMS file format version 2.0
            Next segment offset (64-bit unsigned integer)
            Raw data offset (64-bit unsigned integer)
            
            The above offsets are relative to end of the LeadIn.
        """
        self.endian = '<' # ToC is always little-endian
        self._set_dtypes()
        self.TDSm_tag = self.read_numeric(self.LVstring1, c=4).tobytes().decode(self.encoding)
        
        if self.EOD():
            pass
        elif self.TDSm_tag != 'TDSm':
            raise ValueError(f'Wrong "TDSm" tag value ({self.TDSm_tag}).')
        else:
            self.ToC_mask = self.read_numeric(self.LVuint32)
            self.VersionNumber = self.read_numeric(self.LVuint32)
            self.RemSegLen = self.read_numeric(self.LVuint64)
            if self.RemSegLen == 0xFFFFFFFF:
                print('There was a severe problem while writing to the TDMS file.')
            self.MetaLen = self.read_numeric(self.LVuint64)
            self._segments.append((self.fobj.tell(),
                                   self.ToC_mask, self.VersionNumber,
                                   self.RemSegLen, self.MetaLen,
                                   ))
        
        return not self.EOD()

    def check_ToC_mask(self):
        """Check ToC_mask.
        
        See kToc* in __init__.
        """
        if self.ToC_mask & self.kTocMetaData:
            print('Segment contains MetaData')
        if self.ToC_mask & self.kTocNewObjList:
            print('Segment contains NewObjList')
        if self.ToC_mask & self.kTocRawData:
            print('Segment contains RawData')
        if self.ToC_mask & self.kTocInterleavedData:
            print('Segment contains InterleavedData')
        if self.ToC_mask & self.kTocBigEndian:
            print('Numeric values are BigEndian')
        if self.ToC_mask & self.kTocDAQmxRawData:
            print('Segment contains DAQmxRawData')

    def read_MetaData(self):
        """...
        
        Meta Data layout:
            Number of new objects in this segment (unsigned 32-bit integer)
            Object path (string)
            Raw data index
        """
        print('Segment contains MetaData')
        self.NewObj = self.read_numeric(self.LVuint32)
        print('NewObj', self.NewObj)
        
        for v in range(self.NewObj):
            self.ObjPath = self.read_string()
            print('ObjPath', self.ObjPath)
            self.RDI = self.read_numeric(self.LVuint32)
            print('RDI', self.RDI)
#            if self.RDI == 0xFFFFFFFF:
#                print('object does not have any raw data assigned to it in this segment')
#            elif self.RDI == 0x00001269:
#                print('DAQmx Format Changing scaler')
#            elif self.RDI == 0x00001369:
#                print('DAQmx Digital Line scaler')
#            elif self.RDI == 0x00000000:
#                print('raw data index of this object in this segment exactly matches')
#            else:
#                print('Read raw data')
#                self.read_RawData()
            self.NumProps = self.read_numeric(self.LVuint32)
            print('NumProps', self.NumProps)
            
            for v in range(self.NumProps):
                self.Name = self.read_string()
                self.DataType = self.read_numeric(self.LVuint32)
                
                if self.DataType == self.tdsTypeI8:
                    self.Value = self.read_numeric(self.LVint8)
                if self.DataType == self.tdsTypeI16:
                    self.Value = self.read_numeric(self.LVint16)
                if self.DataType == self.tdsTypeI32:
                    self.Value = self.read_numeric(self.LVint32)
                if self.DataType == self.tdsTypeI64:
                    self.Value = self.read_numeric(self.LVint64)
                if self.DataType == self.tdsTypeU8:
                    self.Value = self.read_numeric(self.LVuint8)
                if self.DataType == self.tdsTypeU16:
                    self.Value = self.read_numeric(self.LVuint16)
                if self.DataType == self.tdsTypeU32:
                    self.Value = self.read_numeric(self.LVuint32)
                if self.DataType == self.tdsTypeU64:
                    self.Value = self.read_numeric(self.LVuint64)
                if self.DataType == self.tdsTypeString:
                    self.Value = self.read_string()
                if self.DataType == self.tdsTypeBoolean:
                    self.Value = self.read_boolean()
                if self.DataType == self.tdsTypeTimeStamp:
                    self.Value = self.read_timestamp()
                self.Props.append((self.Name[0], self.DataType, self.Value))
#                print('\tProp', v, self.Name, self.DataType, self.Value)

    def read_RawData(self):
        """...
        
        Raw Data layout:
        """
        print('Segment contains RawData')
        self.RDIlen = self.read_numeric(self.LVuint32)
        print('RDIlen', self.RDIlen)
        self.DataType = self.read_numeric(self.LVuint32)
        print('DataType', self.DataType)
        self.ArrayDim = self.read_numeric(self.LVuint32)
        print('ArrayDim', self.ArrayDim)
        self.NumValues = self.read_numeric(self.LVuint64)
        print('NumValues', self.NumValues)
        
        if self.DataType == self.tdsTypeString:
            self.Value = self.read_string()
            print(self.Value)
#        print('\tProp', v, self.Name, self.DataType, self.Value)

    def EOD(self):
        """End-Of-Data.
        
        Function overloaded from the base class to make use of the TDSm_tag
        attribute.
        """
        return self.TDSm_tag == ''
