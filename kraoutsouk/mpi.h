/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
*  (C) 2001 by Argonne National Laboratory.
*      See COPYRIGHT in top-level directory.
*/
/* @configure_input@ */
#ifndef MPI_INCLUDED
#define MPI_INCLUDED

/* user include file for MPI programs */

/* Keep C++ compilers from getting confused */
#if defined(__cplusplus)
extern "C" {
#endif

#define MPIAPI __stdcall

	/* Results of the compare operations. */
#define MPI_IDENT     0
#define MPI_CONGRUENT 1
#define MPI_SIMILAR   2
#define MPI_UNEQUAL   3

	typedef int MPI_Datatype;
#define MPI_CHAR           ((MPI_Datatype)0x4c000101)
#define MPI_SIGNED_CHAR    ((MPI_Datatype)0x4c000118)
#define MPI_UNSIGNED_CHAR  ((MPI_Datatype)0x4c000102)
#define MPI_BYTE           ((MPI_Datatype)0x4c00010d)
#define MPI_WCHAR          ((MPI_Datatype)0x4c00020e)
#define MPI_SHORT          ((MPI_Datatype)0x4c000203)
#define MPI_UNSIGNED_SHORT ((MPI_Datatype)0x4c000204)
#define MPI_INT            ((MPI_Datatype)0x4c000405)
#define MPI_UNSIGNED       ((MPI_Datatype)0x4c000406)
#define MPI_LONG           ((MPI_Datatype)0x4c000407)
#define MPI_UNSIGNED_LONG  ((MPI_Datatype)0x4c000408)
#define MPI_FLOAT          ((MPI_Datatype)0x4c00040a)
#define MPI_DOUBLE         ((MPI_Datatype)0x4c00080b)
#define MPI_LONG_DOUBLE    ((MPI_Datatype)0x4c00080c)
#define MPI_LONG_LONG_INT  ((MPI_Datatype)0x4c000809)
#define MPI_UNSIGNED_LONG_LONG ((MPI_Datatype)0x4c000819)
#define MPI_LONG_LONG      MPI_LONG_LONG_INT

#define MPI_PACKED         ((MPI_Datatype)0x4c00010f)
#define MPI_LB             ((MPI_Datatype)0x4c000010)
#define MPI_UB             ((MPI_Datatype)0x4c000011)

	/*
	The layouts for the types MPI_DOUBLE_INT etc are simply
	struct {
	double var;
	int    loc;
	}
	This is documented in the man pages on the various datatypes.
	*/
#define MPI_FLOAT_INT         ((MPI_Datatype)0x8c000000)
#define MPI_DOUBLE_INT        ((MPI_Datatype)0x8c000001)
#define MPI_LONG_INT          ((MPI_Datatype)0x8c000002)
#define MPI_SHORT_INT         ((MPI_Datatype)0x8c000003)
#define MPI_2INT              ((MPI_Datatype)0x4c000816)
#define MPI_LONG_DOUBLE_INT   ((MPI_Datatype)0x8c000004)

	/* Fortran types */
#define MPI_COMPLEX           ((MPI_Datatype)0x4c00081e)
#define MPI_DOUBLE_COMPLEX    ((MPI_Datatype)0x4c001022)
#define MPI_LOGICAL           ((MPI_Datatype)0x4c00041d)
#define MPI_REAL              ((MPI_Datatype)0x4c00041c)
#define MPI_DOUBLE_PRECISION  ((MPI_Datatype)0x4c00081f)
#define MPI_INTEGER           ((MPI_Datatype)0x4c00041b)
#define MPI_2INTEGER          ((MPI_Datatype)0x4c000820)
#define MPI_2COMPLEX          ((MPI_Datatype)0x4c001024)
#define MPI_2DOUBLE_COMPLEX   ((MPI_Datatype)0x4c002025)
#define MPI_2REAL             ((MPI_Datatype)0x4c000821)
#define MPI_2DOUBLE_PRECISION ((MPI_Datatype)0x4c001023)
#define MPI_CHARACTER         ((MPI_Datatype)0x4c00011a)

	/* Size-specific types (see MPI-2, 10.2.5) */
#define MPI_REAL4             ((MPI_Datatype)0x4c000427)
#define MPI_REAL8             ((MPI_Datatype)0x4c000829)
#define MPI_REAL16            ((MPI_Datatype)0x4c00102b)
#define MPI_COMPLEX8          ((MPI_Datatype)0x4c000828)
#define MPI_COMPLEX16         ((MPI_Datatype)0x4c00102a)
#define MPI_COMPLEX32         ((MPI_Datatype)0x4c00202c)
#define MPI_INTEGER1          ((MPI_Datatype)0x4c00012d)
#define MPI_INTEGER2          ((MPI_Datatype)0x4c00022f)
#define MPI_INTEGER4          ((MPI_Datatype)0x4c000430)
#define MPI_INTEGER8          ((MPI_Datatype)0x4c000831)
#define MPI_INTEGER16         ((MPI_Datatype)0x4c001032)

	/* typeclasses */
#define MPI_TYPECLASS_REAL 1
#define MPI_TYPECLASS_INTEGER 2
#define MPI_TYPECLASS_COMPLEX 3

	/* Communicators */
	typedef int MPI_Comm;
#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)
#define MPI_COMM_SELF  ((MPI_Comm)0x44000001)

	/* Groups */
	typedef int MPI_Group;
#define MPI_GROUP_EMPTY ((MPI_Group)0x48000000)

	/* RMA and Windows */
	typedef int MPI_Win;
#define MPI_WIN_NULL ((MPI_Win)0x20000000)

	/* File and IO */
	/* This define lets ROMIO know that MPI_File has been defined */
#define MPI_FILE_DEFINED
	/* ROMIO uses a pointer for MPI_File objects.  This must be the same definition
	as in src/mpi/romio/include/mpio.h.in  */
	typedef struct ADIOI_FileD *MPI_File;
#define MPI_FILE_NULL ((MPI_File)0)

	/* Collective operations */
	typedef int MPI_Op;

#define MPI_MAX     (MPI_Op)(0x58000001)
#define MPI_MIN     (MPI_Op)(0x58000002)
#define MPI_SUM     (MPI_Op)(0x58000003)
#define MPI_PROD    (MPI_Op)(0x58000004)
#define MPI_LAND    (MPI_Op)(0x58000005)
#define MPI_BAND    (MPI_Op)(0x58000006)
#define MPI_LOR     (MPI_Op)(0x58000007)
#define MPI_BOR     (MPI_Op)(0x58000008)
#define MPI_LXOR    (MPI_Op)(0x58000009)
#define MPI_BXOR    (MPI_Op)(0x5800000a)
#define MPI_MINLOC  (MPI_Op)(0x5800000b)
#define MPI_MAXLOC  (MPI_Op)(0x5800000c)
#define MPI_REPLACE (MPI_Op)(0x5800000d)

	/* Permanent key values */
	/* C Versions (return pointer to value),
	Fortran Versions (return integer value).
	Handled directly by the attribute value routine

	DO NOT CHANGE THESE.  The values encode:
	builtin kind (0x1 in bit 30-31)
	Keyval object (0x9 in bits 26-29)
	for communicator (0x1 in bits 22-25)

	Fortran versions of the attributes are formed by adding one to
	the C version.
	*/
#define MPI_TAG_UB           0x64400001
#define MPI_HOST             0x64400003
#define MPI_IO               0x64400005
#define MPI_WTIME_IS_GLOBAL  0x64400007
#define MPI_UNIVERSE_SIZE    0x64400009
#define MPI_LASTUSEDCODE     0x6440000b
#define MPI_APPNUM           0x6440000d

	/* In addition, there are 3 predefined window attributes that are
	defined for every window */
#define MPI_WIN_BASE         0x66000001
#define MPI_WIN_SIZE         0x66000003
#define MPI_WIN_DISP_UNIT    0x66000005

	/* Define some null objects */
#define MPI_COMM_NULL      ((MPI_Comm)0x04000000)
#define MPI_OP_NULL        ((MPI_Op)0x18000000)
#define MPI_GROUP_NULL     ((MPI_Group)0x08000000)
#define MPI_DATATYPE_NULL  ((MPI_Datatype)0x0c000000)
#define MPI_REQUEST_NULL   ((MPI_Request)0x2c000000)
#define MPI_ERRHANDLER_NULL ((MPI_Errhandler)0x14000000)

	/* These are only guesses; make sure you change them in mpif.h as well */
#define MPI_MAX_PROCESSOR_NAME 128
#define MPI_MAX_ERROR_STRING   512 
#define MPI_MAX_NAME_STRING     63
#define MPI_MAX_PORT_NAME      256
#define MPI_MAX_OBJECT_NAME    128

	/* Pre-defined constants */
#define MPI_UNDEFINED      (-32766)
#define MPI_UNDEFINED_RANK MPI_UNDEFINED
#define MPI_KEYVAL_INVALID 0x24000000

	/* Upper bound on the overhead in bsend for each message buffer */
#define MPI_BSEND_OVERHEAD (95)

	/* Topology types */
	typedef enum MPIR_Topo_type { MPI_GRAPH = 1, MPI_CART = 2 } MPIR_Topo_type;

#define MPI_BOTTOM      (void *)0

#define MPI_PROC_NULL   (-1)
#define MPI_ANY_SOURCE 	(-2)
#define MPI_ROOT        (-3)
#define MPI_ANY_TAG     (-1)

#define MPI_LOCK_EXCLUSIVE  234
#define MPI_LOCK_SHARED     235

	/* C functions */
	typedef void (MPIAPI MPI_Handler_function) (MPI_Comm *, int *, ...);
	typedef int (MPIAPI MPI_Comm_copy_attr_function)(MPI_Comm, int, void *, void *,
		void *, int *);
	typedef int (MPIAPI MPI_Comm_delete_attr_function)(MPI_Comm, int, void *, void *);
	typedef int (MPIAPI MPI_Type_copy_attr_function)(MPI_Datatype, int, void *, void *,
		void *, int *);
	typedef int (MPIAPI MPI_Type_delete_attr_function)(MPI_Datatype, int, void *, void *);
	typedef int (MPIAPI MPI_Win_copy_attr_function)(MPI_Win, int, void *, void *, void *,
		int *);
	typedef int (MPIAPI MPI_Win_delete_attr_function)(MPI_Win, int, void *, void *);
	typedef void (MPIAPI MPI_Comm_errhandler_fn)(MPI_Comm *, int *, ...);
	typedef void (MPIAPI MPI_File_errhandler_fn)(MPI_File *, int *, ...);
	typedef void (MPIAPI MPI_Win_errhandler_fn)(MPI_Win *, int *, ...);

	/* Built in (0x1 in 30-31), errhandler (0x5 in bits 26-29, allkind (0
	in 22-25), index in the low bits */
#define MPI_ERRORS_ARE_FATAL ((MPI_Errhandler)0x54000000)
#define MPI_ERRORS_RETURN    ((MPI_Errhandler)0x54000001)
	typedef int MPI_Errhandler;

	/* Make the C names for the dup function mixed case.
	This is required for systems that use all uppercase names for Fortran
	externals.  */
	/* MPI 1 names */
#define MPI_NULL_COPY_FN   ((MPI_Copy_function *)0)
#define MPI_NULL_DELETE_FN ((MPI_Delete_function *)0)
#define MPI_DUP_FN         MPIR_Dup_fn
	/* MPI 2 names */
#define MPI_COMM_NULL_COPY_FN ((MPI_Comm_copy_attr_function*)0)
#define MPI_COMM_NULL_DELETE_FN ((MPI_Comm_delete_attr_function*)0)
#define MPI_COMM_DUP_FN  ((MPI_Comm_copy_attr_function *)MPI_DUP_FN)
#define MPI_WIN_NULL_COPY_FN ((MPI_Win_copy_attr_function*)0)
#define MPI_WIN_NULL_DELETE_FN ((MPI_Win_delete_attr_function*)0)
#define MPI_WIN_DUP_FN   ((MPI_Win_copy_attr_function*)MPI_DUP_FN)
#define MPI_TYPE_NULL_COPY_FN ((MPI_Type_copy_attr_function*)0)
#define MPI_TYPE_NULL_DELETE_FN ((MPI_Type_delete_attr_function*)0)
#define MPI_TYPE_DUP_FN ((MPI_Type_copy_attr_function*)MPI_DUP_FN)

	/* MPI request opjects */
	typedef int MPI_Request;

	/* User combination function */
	typedef void (MPIAPI MPI_User_function) (void *, void *, int *, MPI_Datatype *);

	/* MPI Attribute copy and delete functions */
	typedef int (MPIAPI MPI_Copy_function) (MPI_Comm, int, void *, void *, void *, int *);
	typedef int (MPIAPI MPI_Delete_function) (MPI_Comm, int, void *, void *);

#define MPI_VERSION    2
#define MPI_SUBVERSION 0
#define MPICH_NAME     2
#define MPICH2         1
#define MPICH_HAS_C2F  1
	/* MPICH2_VERSION is the version string */
	/* MPICH2_NUMVERSION is the numeric version that can be used in
	numeric comparisons, such as MPICH2_NUMVERSION < 10005
	This is 1.xx.xx */
#define MPICH2_VERSION "1.0.6"
#define MPICH2_NUMVERSION 10006

	/* for the datatype decoders */
	enum MPI_COMBINER_ENUM {
		MPI_COMBINER_NAMED = 1,
		MPI_COMBINER_DUP = 2,
		MPI_COMBINER_CONTIGUOUS = 3,
		MPI_COMBINER_VECTOR = 4,
		MPI_COMBINER_HVECTOR_INTEGER = 5,
		MPI_COMBINER_HVECTOR = 6,
		MPI_COMBINER_INDEXED = 7,
		MPI_COMBINER_HINDEXED_INTEGER = 8,
		MPI_COMBINER_HINDEXED = 9,
		MPI_COMBINER_INDEXED_BLOCK = 10,
		MPI_COMBINER_STRUCT_INTEGER = 11,
		MPI_COMBINER_STRUCT = 12,
		MPI_COMBINER_SUBARRAY = 13,
		MPI_COMBINER_DARRAY = 14,
		MPI_COMBINER_F90_REAL = 15,
		MPI_COMBINER_F90_COMPLEX = 16,
		MPI_COMBINER_F90_INTEGER = 17,
		MPI_COMBINER_RESIZED = 18
	};

	/* for info */
	typedef int MPI_Info;
#define MPI_INFO_NULL         ((MPI_Info)0x1c000000)
#define MPI_MAX_INFO_KEY       255
#define MPI_MAX_INFO_VAL      1024

	/* for subarray and darray constructors */
#define MPI_ORDER_C              56
#define MPI_ORDER_FORTRAN        57
#define MPI_DISTRIBUTE_BLOCK    121
#define MPI_DISTRIBUTE_CYCLIC   122
#define MPI_DISTRIBUTE_NONE     123
#define MPI_DISTRIBUTE_DFLT_DARG -49767

#define MPI_IN_PLACE  (void *) -1

	/* asserts for one-sided communication */
#define MPI_MODE_NOCHECK      1024
#define MPI_MODE_NOSTORE      2048
#define MPI_MODE_NOPUT        4096
#define MPI_MODE_NOPRECEDE    8192
#define MPI_MODE_NOSUCCEED   16384 

	/* Definitions that are determined by configure. */

#ifdef _WIN64
	typedef __int64 MPI_Aint;
#else
	typedef int MPI_Aint;
#endif

	typedef int MPI_Fint;

	/* Let ROMIO know that MPI_Offset is already defined */
#define HAVE_MPI_OFFSET
	typedef __int64 MPI_Offset;

	/* The order of these elements must match that in mpif.h */
	typedef struct MPI_Status {
		int count;
		int cancelled;
		int MPI_SOURCE;
		int MPI_TAG;
		int MPI_ERROR;

	} MPI_Status;

	/* Handle conversion types/functions */

	/* Programs that need to convert types used in MPICH should use these */
#define MPI_Comm_c2f(comm) (MPI_Fint)(comm)
#define MPI_Comm_f2c(comm) (MPI_Comm)(comm)
#define MPI_Type_c2f(datatype) (MPI_Fint)(datatype)
#define MPI_Type_f2c(datatype) (MPI_Datatype)(datatype)
#define MPI_Group_c2f(group) (MPI_Fint)(group)
#define MPI_Group_f2c(group) (MPI_Group)(group)
#define MPI_Info_c2f(info) (MPI_Fint)(info)
#define MPI_Info_f2c(info) (MPI_Info)(info)
#define MPI_Request_f2c(request) (MPI_Request)(request)
#define MPI_Request_c2f(request) (MPI_Fint)(request)
#define MPI_Op_c2f(op) (MPI_Fint)(op)
#define MPI_Op_f2c(op) (MPI_Op)(op)
#define MPI_Errhandler_c2f(errhandler) (MPI_Fint)(errhandler)
#define MPI_Errhandler_f2c(errhandler) (MPI_Errhandler)(errhandler)
#define MPI_Win_c2f(win)   (MPI_Fint)(win)
#define MPI_Win_f2c(win)   (MPI_Win)(win)

	/* PMPI versions of the handle transfer functions.  See section 4.17 */
#define PMPI_Comm_c2f(comm) (MPI_Fint)(comm)
#define PMPI_Comm_f2c(comm) (MPI_Comm)(comm)
#define PMPI_Type_c2f(datatype) (MPI_Fint)(datatype)
#define PMPI_Type_f2c(datatype) (MPI_Datatype)(datatype)
#define PMPI_Group_c2f(group) (MPI_Fint)(group)
#define PMPI_Group_f2c(group) (MPI_Group)(group)
#define PMPI_Info_c2f(info) (MPI_Fint)(info)
#define PMPI_Info_f2c(info) (MPI_Info)(info)
#define PMPI_Request_f2c(request) (MPI_Request)(request)
#define PMPI_Request_c2f(request) (MPI_Fint)(request)
#define PMPI_Op_c2f(op) (MPI_Fint)(op)
#define PMPI_Op_f2c(op) (MPI_Op)(op)
#define PMPI_Errhandler_c2f(errhandler) (MPI_Fint)(errhandler)
#define PMPI_Errhandler_f2c(errhandler) (MPI_Errhandler)(errhandler)
#define PMPI_Win_c2f(win)   (MPI_Fint)(win)
#define PMPI_Win_f2c(win)   (MPI_Win)(win)

#define MPI_STATUS_IGNORE (MPI_Status *)1
#define MPI_STATUSES_IGNORE (MPI_Status *)1
#define MPI_ERRCODES_IGNORE (int *)0

	/* See 4.12.5 for MPI_F_STATUS(ES)_IGNORE */
#if !defined(_MPICH_DLL_)
#define MPIU_DLL_SPEC __declspec(dllimport)
#else
#define MPIU_DLL_SPEC
#endif

	extern MPIU_DLL_SPEC MPI_Fint * MPI_F_STATUS_IGNORE;
	extern MPIU_DLL_SPEC MPI_Fint * MPI_F_STATUSES_IGNORE;

	/* The MPI standard requires that the ARGV_NULL values be the same as NULL (see 5.3.2) */
#define MPI_ARGV_NULL (char **)0
#define MPI_ARGVS_NULL (char ***)0

	/* For supported thread levels */
#define MPI_THREAD_SINGLE 0
#define MPI_THREAD_FUNNELED 1
#define MPI_THREAD_SERIALIZED 2
#define MPI_THREAD_MULTIPLE 3

	/* Typedefs for generalized requests */
	typedef int (MPIAPI MPI_Grequest_cancel_function)(void *, int);
	typedef int (MPIAPI MPI_Grequest_free_function)(void *);
	typedef int (MPIAPI MPI_Grequest_query_function)(void *, MPI_Status *);

	/* MPI's error classes */
#define MPI_SUCCESS          0      /* Successful return code */
	/* Communication argument parameters */
#define MPI_ERR_BUFFER       1      /* Invalid buffer pointer */
#define MPI_ERR_COUNT        2      /* Invalid count argument */
#define MPI_ERR_TYPE         3      /* Invalid datatype argument */
#define MPI_ERR_TAG          4      /* Invalid tag argument */
#define MPI_ERR_COMM         5      /* Invalid communicator */
#define MPI_ERR_RANK         6      /* Invalid rank */
#define MPI_ERR_ROOT         7      /* Invalid root */
#define MPI_ERR_TRUNCATE    14      /* Message truncated on receive */

	/* MPI Objects (other than COMM) */
#define MPI_ERR_GROUP        8      /* Invalid group */
#define MPI_ERR_OP           9      /* Invalid operation */
#define MPI_ERR_REQUEST     19      /* Invalid mpi_request handle */

	/* Special topology argument parameters */
#define MPI_ERR_TOPOLOGY    10      /* Invalid topology */
#define MPI_ERR_DIMS        11      /* Invalid dimension argument */

	/* All other arguments.  This is a class with many kinds */
#define MPI_ERR_ARG         12      /* Invalid argument */

	/* Other errors that are not simply an invalid argument */
#define MPI_ERR_OTHER       15      /* Other error; use Error_string */

#define MPI_ERR_UNKNOWN     13      /* Unknown error */
#define MPI_ERR_INTERN      16      /* Internal error code    */

	/* Multiple completion has two special error classes */
#define MPI_ERR_IN_STATUS   17      /* Look in status for error value */
#define MPI_ERR_PENDING     18      /* Pending request */

	/* New MPI-2 Error classes */
#define MPI_ERR_FILE        27      /* */
#define MPI_ERR_ACCESS      20      /* */
#define MPI_ERR_AMODE       21      /* */
#define MPI_ERR_BAD_FILE    22      /* */
#define MPI_ERR_FILE_EXISTS 25      /* */
#define MPI_ERR_FILE_IN_USE 26      /* */
#define MPI_ERR_NO_SPACE    36      /* */
#define MPI_ERR_NO_SUCH_FILE 37     /* */
#define MPI_ERR_IO          32      /* */
#define MPI_ERR_READ_ONLY   40      /* */
#define MPI_ERR_CONVERSION  23      /* */
#define MPI_ERR_DUP_DATAREP 24      /* */
#define MPI_ERR_UNSUPPORTED_DATAREP   43  /* */

	/* MPI_ERR_INFO is NOT defined in the MPI-2 standard.  I believe that
	this is an oversight */
#define MPI_ERR_INFO        28      /* */
#define MPI_ERR_INFO_KEY    29      /* */
#define MPI_ERR_INFO_VALUE  30      /* */
#define MPI_ERR_INFO_NOKEY  31      /* */

#define MPI_ERR_NAME        33      /* */
#define MPI_ERR_NO_MEM      34      /* Alloc_mem could not allocate memory */
#define MPI_ERR_NOT_SAME    35      /* */
#define MPI_ERR_PORT        38      /* */
#define MPI_ERR_QUOTA       39      /* */
#define MPI_ERR_SERVICE     41      /* */
#define MPI_ERR_SPAWN       42      /* */
#define MPI_ERR_UNSUPPORTED_OPERATION 44 /* */
#define MPI_ERR_WIN         45      /* */

#define MPI_ERR_BASE        46      /* */
#define MPI_ERR_LOCKTYPE    47      /* */
#define MPI_ERR_KEYVAL      48      /* Erroneous attribute key */
#define MPI_ERR_RMA_CONFLICT 49     /* */
#define MPI_ERR_RMA_SYNC    50      /* */ 
#define MPI_ERR_SIZE        51      /* */
#define MPI_ERR_DISP        52      /* */
#define MPI_ERR_ASSERT      53      /* */

#define MPI_ERR_LASTCODE    0x3fffffff  /* Last valid error code for a 
	predefined error class */
#define MPICH_ERR_LAST_CLASS 53     /* It is also helpful to know the
	last valid class */
	/* End of MPI's error classes */

	/* Function type defs */
	typedef int (MPIAPI MPI_Datarep_conversion_function)(void *, MPI_Datatype, int,
		void *, MPI_Offset, void *);
	typedef int (MPIAPI MPI_Datarep_extent_function)(MPI_Datatype datatype, MPI_Aint *,
		void *);
#define MPI_CONVERSION_FN_NULL ((MPI_Datarep_conversion_function *)0)

	/*
	For systems that may need to add additional definitions to support
	different declaration styles and options (e.g., different calling
	conventions or DLL import/export controls).
	*/
	/* --Insert Additional Definitions Here-- */

	/* Include any device specific definitions */
	/* No device specific defs */

	/*
	* Normally, we provide prototypes for all MPI routines.  In a few wierd
	* cases, we need to suppress the prototypes.
	*/
#ifndef MPICH_SUPPRESS_PROTOTYPES
	/* We require that the C compiler support prototypes */
	/* Begin Prototypes */
	int MPIAPI MPI_Send(void*, int, MPI_Datatype, int, int, MPI_Comm);
	int MPIAPI MPI_Recv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
	int MPIAPI MPI_Get_count(MPI_Status *, MPI_Datatype, int *);
	int MPIAPI MPI_Bsend(void*, int, MPI_Datatype, int, int, MPI_Comm);
	int MPIAPI MPI_Ssend(void*, int, MPI_Datatype, int, int, MPI_Comm);
	int MPIAPI MPI_Rsend(void*, int, MPI_Datatype, int, int, MPI_Comm);
	int MPIAPI MPI_Buffer_attach(void*, int);
	int MPIAPI MPI_Buffer_detach(void*, int *);
	int MPIAPI MPI_Isend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI MPI_Ibsend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI MPI_Issend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI MPI_Irsend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI MPI_Irecv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI MPI_Wait(MPI_Request *, MPI_Status *);
	int MPIAPI MPI_Test(MPI_Request *, int *, MPI_Status *);
	int MPIAPI MPI_Request_free(MPI_Request *);
	int MPIAPI MPI_Waitany(int, MPI_Request *, int *, MPI_Status *);
	int MPIAPI MPI_Testany(int, MPI_Request *, int *, int *, MPI_Status *);
	int MPIAPI MPI_Waitall(int, MPI_Request *, MPI_Status *);
	int MPIAPI MPI_Testall(int, MPI_Request *, int *, MPI_Status *);
	int MPIAPI MPI_Waitsome(int, MPI_Request *, int *, int *, MPI_Status *);
	int MPIAPI MPI_Testsome(int, MPI_Request *, int *, int *, MPI_Status *);
	int MPIAPI MPI_Iprobe(int, int, MPI_Comm, int *, MPI_Status *);
	int MPIAPI MPI_Probe(int, int, MPI_Comm, MPI_Status *);
	int MPIAPI MPI_Cancel(MPI_Request *);
	int MPIAPI MPI_Test_cancelled(MPI_Status *, int *);
	int MPIAPI MPI_Send_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI MPI_Bsend_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI MPI_Ssend_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI MPI_Rsend_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI MPI_Recv_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI MPI_Start(MPI_Request *);
	int MPIAPI MPI_Startall(int, MPI_Request *);
	int MPIAPI MPI_Sendrecv(void *, int, MPI_Datatype, int, int, void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
	int MPIAPI MPI_Sendrecv_replace(void*, int, MPI_Datatype, int, int, int, int, MPI_Comm, MPI_Status *);
	int MPIAPI MPI_Type_contiguous(int, MPI_Datatype, MPI_Datatype *);
	int MPIAPI MPI_Type_vector(int, int, int, MPI_Datatype, MPI_Datatype *);
	int MPIAPI MPI_Type_hvector(int, int, MPI_Aint, MPI_Datatype, MPI_Datatype *);
	int MPIAPI MPI_Type_indexed(int, int *, int *, MPI_Datatype, MPI_Datatype *);
	int MPIAPI MPI_Type_hindexed(int, int *, MPI_Aint *, MPI_Datatype, MPI_Datatype *);
	int MPIAPI MPI_Type_struct(int, int *, MPI_Aint *, MPI_Datatype *, MPI_Datatype *);
	int MPIAPI MPI_Address(void*, MPI_Aint *);
	/* We could add __attribute__((deprecated)) to routines like MPI_Type_extent */
	int MPIAPI MPI_Type_extent(MPI_Datatype, MPI_Aint *);
	/* See the 1.1 version of the Standard.  The standard made an
	unfortunate choice here, however, it is the standard.  The size returned
	by MPI_Type_size is specified as an int, not an MPI_Aint */
	int MPIAPI MPI_Type_size(MPI_Datatype, int *);
	/* MPI_Type_count was withdrawn in MPI 1.1 */
	int MPIAPI MPI_Type_lb(MPI_Datatype, MPI_Aint *);
	int MPIAPI MPI_Type_ub(MPI_Datatype, MPI_Aint *);
	int MPIAPI MPI_Type_commit(MPI_Datatype *);
	int MPIAPI MPI_Type_free(MPI_Datatype *);
	int MPIAPI MPI_Get_elements(MPI_Status *, MPI_Datatype, int *);
	int MPIAPI MPI_Pack(void*, int, MPI_Datatype, void *, int, int *, MPI_Comm);
	int MPIAPI MPI_Unpack(void*, int, int *, void *, int, MPI_Datatype, MPI_Comm);
	int MPIAPI MPI_Pack_size(int, MPI_Datatype, MPI_Comm, int *);
	int MPIAPI MPI_Barrier(MPI_Comm);
	int MPIAPI MPI_Bcast(void*, int, MPI_Datatype, int, MPI_Comm);
	int MPIAPI MPI_Gather(void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm);
	int MPIAPI MPI_Gatherv(void*, int, MPI_Datatype, void*, int *, int *, MPI_Datatype, int, MPI_Comm);
	int MPIAPI MPI_Scatter(void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm);
	int MPIAPI MPI_Scatterv(void*, int *, int *, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm);
	int MPIAPI MPI_Allgather(void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm);
	int MPIAPI MPI_Allgatherv(void*, int, MPI_Datatype, void*, int *, int *, MPI_Datatype, MPI_Comm);
	int MPIAPI MPI_Alltoall(void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm);
	int MPIAPI MPI_Alltoallv(void*, int *, int *, MPI_Datatype, void*, int *, int *, MPI_Datatype, MPI_Comm);
	int MPIAPI MPI_Reduce(void*, void*, int, MPI_Datatype, MPI_Op, int, MPI_Comm);
	int MPIAPI MPI_Op_create(MPI_User_function *, int, MPI_Op *);
	int MPIAPI MPI_Op_free(MPI_Op *);
	int MPIAPI MPI_Allreduce(void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
	int MPIAPI MPI_Reduce_scatter(void*, void*, int *, MPI_Datatype, MPI_Op, MPI_Comm);
	int MPIAPI MPI_Scan(void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
	int MPIAPI MPI_Group_size(MPI_Group, int *);
	int MPIAPI MPI_Group_rank(MPI_Group, int *);
	int MPIAPI MPI_Group_translate_ranks(MPI_Group, int, int *, MPI_Group, int *);
	int MPIAPI MPI_Group_compare(MPI_Group, MPI_Group, int *);
	int MPIAPI MPI_Comm_group(MPI_Comm, MPI_Group *);
	int MPIAPI MPI_Group_union(MPI_Group, MPI_Group, MPI_Group *);
	int MPIAPI MPI_Group_intersection(MPI_Group, MPI_Group, MPI_Group *);
	int MPIAPI MPI_Group_difference(MPI_Group, MPI_Group, MPI_Group *);
	int MPIAPI MPI_Group_incl(MPI_Group, int, int *, MPI_Group *);
	int MPIAPI MPI_Group_excl(MPI_Group, int, int *, MPI_Group *);
	int MPIAPI MPI_Group_range_incl(MPI_Group, int, int[][3], MPI_Group *);
	int MPIAPI MPI_Group_range_excl(MPI_Group, int, int[][3], MPI_Group *);
	int MPIAPI MPI_Group_free(MPI_Group *);
	int MPIAPI MPI_Comm_size(MPI_Comm, int *);
	int MPIAPI MPI_Comm_rank(MPI_Comm, int *);
	int MPIAPI MPI_Comm_compare(MPI_Comm, MPI_Comm, int *);
	int MPIAPI MPI_Comm_dup(MPI_Comm, MPI_Comm *);
	int MPIAPI MPI_Comm_create(MPI_Comm, MPI_Group, MPI_Comm *);
	int MPIAPI MPI_Comm_split(MPI_Comm, int, int, MPI_Comm *);
	int MPIAPI MPI_Comm_free(MPI_Comm *);
	int MPIAPI MPI_Comm_test_inter(MPI_Comm, int *);
	int MPIAPI MPI_Comm_remote_size(MPI_Comm, int *);
	int MPIAPI MPI_Comm_remote_group(MPI_Comm, MPI_Group *);
	int MPIAPI MPI_Intercomm_create(MPI_Comm, int, MPI_Comm, int, int, MPI_Comm *);
	int MPIAPI MPI_Intercomm_merge(MPI_Comm, int, MPI_Comm *);
	int MPIAPI MPI_Keyval_create(MPI_Copy_function *, MPI_Delete_function *, int *, void*);
	int MPIAPI MPI_Keyval_free(int *);
	int MPIAPI MPI_Attr_put(MPI_Comm, int, void*);
	int MPIAPI MPI_Attr_get(MPI_Comm, int, void *, int *);
	int MPIAPI MPI_Attr_delete(MPI_Comm, int);
	int MPIAPI MPI_Topo_test(MPI_Comm, int *);
	int MPIAPI MPI_Cart_create(MPI_Comm, int, int *, int *, int, MPI_Comm *);
	int MPIAPI MPI_Dims_create(int, int, int *);
	int MPIAPI MPI_Graph_create(MPI_Comm, int, int *, int *, int, MPI_Comm *);
	int MPIAPI MPI_Graphdims_get(MPI_Comm, int *, int *);
	int MPIAPI MPI_Graph_get(MPI_Comm, int, int, int *, int *);
	int MPIAPI MPI_Cartdim_get(MPI_Comm, int *);
	int MPIAPI MPI_Cart_get(MPI_Comm, int, int *, int *, int *);
	int MPIAPI MPI_Cart_rank(MPI_Comm, int *, int *);
	int MPIAPI MPI_Cart_coords(MPI_Comm, int, int, int *);
	int MPIAPI MPI_Graph_neighbors_count(MPI_Comm, int, int *);
	int MPIAPI MPI_Graph_neighbors(MPI_Comm, int, int, int *);
	int MPIAPI MPI_Cart_shift(MPI_Comm, int, int, int *, int *);
	int MPIAPI MPI_Cart_sub(MPI_Comm, int *, MPI_Comm *);
	int MPIAPI MPI_Cart_map(MPI_Comm, int, int *, int *, int *);
	int MPIAPI MPI_Graph_map(MPI_Comm, int, int *, int *, int *);
	int MPIAPI MPI_Get_processor_name(char *, int *);
	int MPIAPI MPI_Get_version(int *, int *);
	int MPIAPI MPI_Errhandler_create(MPI_Handler_function *, MPI_Errhandler *);
	int MPIAPI MPI_Errhandler_set(MPI_Comm, MPI_Errhandler);
	int MPIAPI MPI_Errhandler_get(MPI_Comm, MPI_Errhandler *);
	int MPIAPI MPI_Errhandler_free(MPI_Errhandler *);
	int MPIAPI MPI_Error_string(int, char *, int *);
	int MPIAPI MPI_Error_class(int, int *);
	double MPIAPI MPI_Wtime(void);
	double MPIAPI MPI_Wtick(void);
#ifndef MPI_Wtime
	double MPIAPI PMPI_Wtime(void);
	double MPIAPI PMPI_Wtick(void);
#endif
	int MPIAPI MPI_Init(int *, char ***);
	int MPIAPI MPI_Finalize(void);
	int MPIAPI MPI_Initialized(int *);
	int MPIAPI MPI_Abort(MPI_Comm, int);


	/* Note that we may need to define a @PCONTROL_LIST@ depending on whether
	stdargs are supported */
	int MPIAPI MPI_Pcontrol(const int, ...);

	int MPIAPI MPI_DUP_FN(MPI_Comm, int, void *, void *, void *, int *);


	/* MPI-2 functions */

	/* Process Creation and Management */
	int MPIAPI MPI_Close_port(char *);
	int MPIAPI MPI_Comm_accept(char *, MPI_Info, int, MPI_Comm, MPI_Comm *);
	int MPIAPI MPI_Comm_connect(char *, MPI_Info, int, MPI_Comm, MPI_Comm *);
	int MPIAPI MPI_Comm_disconnect(MPI_Comm *);
	int MPIAPI MPI_Comm_get_parent(MPI_Comm *);
	int MPIAPI MPI_Comm_join(int, MPI_Comm *);
	int MPIAPI MPI_Comm_spawn(char *, char *[], int, MPI_Info, int, MPI_Comm, MPI_Comm *,
		int[]);
	int MPIAPI MPI_Comm_spawn_multiple(int, char *[], char **[], int[], MPI_Info[], int,
		MPI_Comm, MPI_Comm *, int[]);
	int MPIAPI MPI_Lookup_name(char *, MPI_Info, char *);
	int MPIAPI MPI_Open_port(MPI_Info, char *);
	int MPIAPI MPI_Publish_name(char *, MPI_Info, char *);
	int MPIAPI MPI_Unpublish_name(char *, MPI_Info, char *);

	/* One-Sided Communications */
	int MPIAPI MPI_Accumulate(void *, int, MPI_Datatype, int, MPI_Aint, int,
		MPI_Datatype, MPI_Op, MPI_Win);
	int MPIAPI MPI_Get(void *, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype,
		MPI_Win);
	int MPIAPI MPI_Put(void *, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype,
		MPI_Win);
	int MPIAPI MPI_Win_complete(MPI_Win);
	int MPIAPI MPI_Win_create(void *, MPI_Aint, int, MPI_Info, MPI_Comm, MPI_Win *);
	int MPIAPI MPI_Win_fence(int, MPI_Win);
	int MPIAPI MPI_Win_free(MPI_Win *);
	int MPIAPI MPI_Win_get_group(MPI_Win, MPI_Group *);
	int MPIAPI MPI_Win_lock(int, int, int, MPI_Win);
	int MPIAPI MPI_Win_post(MPI_Group, int, MPI_Win);
	int MPIAPI MPI_Win_start(MPI_Group, int, MPI_Win);
	int MPIAPI MPI_Win_test(MPI_Win, int *);
	int MPIAPI MPI_Win_unlock(int, MPI_Win);
	int MPIAPI MPI_Win_wait(MPI_Win);

	/* Extended Collective Operations */
	int MPIAPI MPI_Alltoallw(void *, int[], int[], MPI_Datatype[], void *, int[],
		int[], MPI_Datatype[], MPI_Comm);
	int MPIAPI MPI_Exscan(void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);

	/* External Interfaces */
	int MPIAPI MPI_Add_error_class(int *);
	int MPIAPI MPI_Add_error_code(int, int *);
	int MPIAPI MPI_Add_error_string(int, char *);
	int MPIAPI MPI_Comm_call_errhandler(MPI_Comm, int);
	int MPIAPI MPI_Comm_create_keyval(MPI_Comm_copy_attr_function *,
		MPI_Comm_delete_attr_function *, int *, void *);
	int MPIAPI MPI_Comm_delete_attr(MPI_Comm, int);
	int MPIAPI MPI_Comm_free_keyval(int *);
	int MPIAPI MPI_Comm_get_attr(MPI_Comm, int, void *, int *);
	int MPIAPI MPI_Comm_get_name(MPI_Comm, char *, int *);
	int MPIAPI MPI_Comm_set_attr(MPI_Comm, int, void *);
	int MPIAPI MPI_Comm_set_name(MPI_Comm, char *);
	int MPIAPI MPI_File_call_errhandler(MPI_File, int);
	int MPIAPI MPI_Grequest_complete(MPI_Request);
	int MPIAPI MPI_Grequest_start(MPI_Grequest_query_function *,
		MPI_Grequest_free_function *,
		MPI_Grequest_cancel_function *, void *, MPI_Request *);
	int MPIAPI MPI_Init_thread(int *, char ***, int, int *);
	int MPIAPI MPI_Is_thread_main(int *);
	int MPIAPI MPI_Query_thread(int *);
	int MPIAPI MPI_Status_set_cancelled(MPI_Status *, int);
	int MPIAPI MPI_Status_set_elements(MPI_Status *, MPI_Datatype, int);
	int MPIAPI MPI_Type_create_keyval(MPI_Type_copy_attr_function *,
		MPI_Type_delete_attr_function *, int *, void *);
	int MPIAPI MPI_Type_delete_attr(MPI_Datatype, int);
	int MPIAPI MPI_Type_dup(MPI_Datatype, MPI_Datatype *);
	int MPIAPI MPI_Type_free_keyval(int *);
	int MPIAPI MPI_Type_get_attr(MPI_Datatype, int, void *, int *);
	int MPIAPI MPI_Type_get_contents(MPI_Datatype, int, int, int, int[], MPI_Aint[],
		MPI_Datatype[]);
	int MPIAPI MPI_Type_get_envelope(MPI_Datatype, int *, int *, int *, int *);
	int MPIAPI MPI_Type_get_name(MPI_Datatype, char *, int *);
	int MPIAPI MPI_Type_set_attr(MPI_Datatype, int, void *);
	int MPIAPI MPI_Type_set_name(MPI_Datatype, char *);
	int MPIAPI MPI_Type_match_size(int, int, MPI_Datatype *);
	int MPIAPI MPI_Win_call_errhandler(MPI_Win, int);
	int MPIAPI MPI_Win_create_keyval(MPI_Win_copy_attr_function *,
		MPI_Win_delete_attr_function *, int *, void *);
	int MPIAPI MPI_Win_delete_attr(MPI_Win, int);
	int MPIAPI MPI_Win_free_keyval(int *);
	int MPIAPI MPI_Win_get_attr(MPI_Win, int, void *, int *);
	int MPIAPI MPI_Win_get_name(MPI_Win, char *, int *);
	int MPIAPI MPI_Win_set_attr(MPI_Win, int, void *);
	int MPIAPI MPI_Win_set_name(MPI_Win, char *);

	/* Miscellany */
#ifdef FOO
	MPI_Comm MPI_Comm_f2c(MPI_Fint);
	MPI_Datatype MPI_Type_f2c(MPI_Fint);
	MPI_File MPI_File_f2c(MPI_Fint);
	MPI_Fint MPIAPI MPI_Comm_c2f(MPI_Comm);
	MPI_Fint MPIAPI MPI_File_c2f(MPI_File);
	MPI_Fint MPI_Group_c2f(MPI_Group);
	MPI_Fint MPI_Info_c2f(MPI_Info);
	MPI_Fint MPI_Op_c2f(MPI_Op);
	MPI_Fint MPI_Request_c2f(MPI_Request);
	MPI_Fint MPI_Type_c2f(MPI_Datatype);
	MPI_Fint MPI_Win_c2f(MPI_Win);
	MPI_Group MPI_Group_f2c(MPI_Fint);
	MPI_Info MPI_Info_f2c(MPI_Fint);
	MPI_Op MPI_Op_f2c(MPI_Fint);
	MPI_Request MPI_Request_f2c(MPI_Fint);
	MPI_Win MPI_Win_f2c(MPI_Fint);
#endif

	int MPIAPI MPI_Alloc_mem(MPI_Aint, MPI_Info info, void *baseptr);
	int MPIAPI MPI_Comm_create_errhandler(MPI_Comm_errhandler_fn *, MPI_Errhandler *);
	int MPIAPI MPI_Comm_get_errhandler(MPI_Comm, MPI_Errhandler *);
	int MPIAPI MPI_Comm_set_errhandler(MPI_Comm, MPI_Errhandler);
	int MPIAPI MPI_File_create_errhandler(MPI_File_errhandler_fn *, MPI_Errhandler *);
	int MPIAPI MPI_File_get_errhandler(MPI_File, MPI_Errhandler *);
	int MPIAPI MPI_File_set_errhandler(MPI_File, MPI_Errhandler);
	int MPIAPI MPI_Finalized(int *);
	int MPIAPI MPI_Free_mem(void *);
	int MPIAPI MPI_Get_address(void *, MPI_Aint *);
	int MPIAPI MPI_Info_create(MPI_Info *);
	int MPIAPI MPI_Info_delete(MPI_Info, char *);
	int MPIAPI MPI_Info_dup(MPI_Info, MPI_Info *);
	int MPIAPI MPI_Info_free(MPI_Info *info);
	int MPIAPI MPI_Info_get(MPI_Info, char *, int, char *, int *);
	int MPIAPI MPI_Info_get_nkeys(MPI_Info, int *);
	int MPIAPI MPI_Info_get_nthkey(MPI_Info, int, char *);
	int MPIAPI MPI_Info_get_valuelen(MPI_Info, char *, int *, int *);
	int MPIAPI MPI_Info_set(MPI_Info, char *, char *);
	int MPIAPI MPI_Pack_external(char *, void *, int, MPI_Datatype, void *, MPI_Aint,
		MPI_Aint *);
	int MPIAPI MPI_Pack_external_size(char *, int, MPI_Datatype, MPI_Aint *);
	int MPIAPI MPI_Request_get_status(MPI_Request, int *, MPI_Status *);
	int MPIAPI MPI_Status_c2f(MPI_Status *, MPI_Fint *);
	int MPIAPI MPI_Status_f2c(MPI_Fint *, MPI_Status *);
	int MPIAPI MPI_Type_create_darray(int, int, int, int[], int[], int[], int[], int,
		MPI_Datatype, MPI_Datatype *);
	int MPIAPI MPI_Type_create_hindexed(int, int[], MPI_Aint[], MPI_Datatype,
		MPI_Datatype *);
	int MPIAPI MPI_Type_create_hvector(int, int, MPI_Aint, MPI_Datatype, MPI_Datatype *);
	int MPIAPI MPI_Type_create_indexed_block(int, int, int[], MPI_Datatype,
		MPI_Datatype *);
	int MPIAPI MPI_Type_create_resized(MPI_Datatype, MPI_Aint, MPI_Aint, MPI_Datatype *);
	int MPIAPI MPI_Type_create_struct(int, int[], MPI_Aint[], MPI_Datatype[],
		MPI_Datatype *);
	int MPIAPI MPI_Type_create_subarray(int, int[], int[], int[], int, MPI_Datatype,
		MPI_Datatype *);
	int MPIAPI MPI_Type_get_extent(MPI_Datatype, MPI_Aint *, MPI_Aint *);
	int MPIAPI MPI_Type_get_true_extent(MPI_Datatype, MPI_Aint *, MPI_Aint *);
	int MPIAPI MPI_Unpack_external(char *, void *, MPI_Aint, MPI_Aint *, void *, int,
		MPI_Datatype);
	int MPIAPI MPI_Win_create_errhandler(MPI_Win_errhandler_fn *, MPI_Errhandler *);
	int MPIAPI MPI_Win_get_errhandler(MPI_Win, MPI_Errhandler *);
	int MPIAPI MPI_Win_set_errhandler(MPI_Win, MPI_Errhandler);

	/* Fortran 90-related functions.  These routines are available only if
	Fortran 90 support is enabled
	*/
	int MPIAPI MPI_Type_create_f90_integer(int, MPI_Datatype *);
	int MPIAPI MPI_Type_create_f90_real(int, int, MPI_Datatype *);
	int MPIAPI MPI_Type_create_f90_complex(int, int, MPI_Datatype *);

	/* End Prototypes */
#endif /* MPICH_SUPPRESS_PROTOTYPES */



	/* Here are the bindings of the profiling routines */
#if !defined(MPI_BUILD_PROFILING)
	int MPIAPI PMPI_Send(void*, int, MPI_Datatype, int, int, MPI_Comm);
	int MPIAPI PMPI_Recv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
	int MPIAPI PMPI_Get_count(MPI_Status *, MPI_Datatype, int *);
	int MPIAPI PMPI_Bsend(void*, int, MPI_Datatype, int, int, MPI_Comm);
	int MPIAPI PMPI_Ssend(void*, int, MPI_Datatype, int, int, MPI_Comm);
	int MPIAPI PMPI_Rsend(void*, int, MPI_Datatype, int, int, MPI_Comm);
	int MPIAPI PMPI_Buffer_attach(void* buffer, int);
	int MPIAPI PMPI_Buffer_detach(void* buffer, int *);
	int MPIAPI PMPI_Isend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI PMPI_Ibsend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI PMPI_Issend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI PMPI_Irsend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI PMPI_Irecv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI PMPI_Wait(MPI_Request *, MPI_Status *);
	int MPIAPI PMPI_Test(MPI_Request *, int *, MPI_Status *);
	int MPIAPI PMPI_Request_free(MPI_Request *);
	int MPIAPI PMPI_Waitany(int, MPI_Request *, int *, MPI_Status *);
	int MPIAPI PMPI_Testany(int, MPI_Request *, int *, int *, MPI_Status *);
	int MPIAPI PMPI_Waitall(int, MPI_Request *, MPI_Status *);
	int MPIAPI PMPI_Testall(int, MPI_Request *, int *, MPI_Status *);
	int MPIAPI PMPI_Waitsome(int, MPI_Request *, int *, int *, MPI_Status *);
	int MPIAPI PMPI_Testsome(int, MPI_Request *, int *, int *, MPI_Status *);
	int MPIAPI PMPI_Iprobe(int, int, MPI_Comm, int *, MPI_Status *);
	int MPIAPI PMPI_Probe(int, int, MPI_Comm, MPI_Status *);
	int MPIAPI PMPI_Cancel(MPI_Request *);
	int MPIAPI PMPI_Test_cancelled(MPI_Status *, int *);
	int MPIAPI PMPI_Send_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI PMPI_Bsend_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI PMPI_Ssend_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI PMPI_Rsend_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI PMPI_Recv_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
	int MPIAPI PMPI_Start(MPI_Request *);
	int MPIAPI PMPI_Startall(int, MPI_Request *);
	int MPIAPI PMPI_Sendrecv(void *, int, MPI_Datatype, int, int, void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
	int MPIAPI PMPI_Sendrecv_replace(void*, int, MPI_Datatype, int, int, int, int, MPI_Comm, MPI_Status *);
	int MPIAPI PMPI_Type_contiguous(int, MPI_Datatype, MPI_Datatype *);
	int MPIAPI PMPI_Type_vector(int, int, int, MPI_Datatype, MPI_Datatype *);
	int MPIAPI PMPI_Type_hvector(int, int, MPI_Aint, MPI_Datatype, MPI_Datatype *);
	int MPIAPI PMPI_Type_indexed(int, int *, int *, MPI_Datatype, MPI_Datatype *);
	int MPIAPI PMPI_Type_hindexed(int, int *, MPI_Aint *, MPI_Datatype, MPI_Datatype *);
	int MPIAPI PMPI_Type_struct(int, int *, MPI_Aint *, MPI_Datatype *, MPI_Datatype *);
	int MPIAPI PMPI_Address(void*, MPI_Aint *);
	int MPIAPI PMPI_Type_extent(MPI_Datatype, MPI_Aint *);
	int MPIAPI PMPI_Type_size(MPI_Datatype, int *);
	int MPIAPI PMPI_Type_lb(MPI_Datatype, MPI_Aint *);
	int MPIAPI PMPI_Type_ub(MPI_Datatype, MPI_Aint *);
	int MPIAPI PMPI_Type_commit(MPI_Datatype *);
	int MPIAPI PMPI_Type_free(MPI_Datatype *);
	int MPIAPI PMPI_Get_elements(MPI_Status *, MPI_Datatype, int *);
	int MPIAPI PMPI_Pack(void*, int, MPI_Datatype, void *, int, int *, MPI_Comm);
	int MPIAPI PMPI_Unpack(void*, int, int *, void *, int, MPI_Datatype, MPI_Comm);
	int MPIAPI PMPI_Pack_size(int, MPI_Datatype, MPI_Comm, int *);
	int MPIAPI PMPI_Barrier(MPI_Comm);
	int MPIAPI PMPI_Bcast(void* buffer, int, MPI_Datatype, int, MPI_Comm);
	int MPIAPI PMPI_Gather(void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm);
	int MPIAPI PMPI_Gatherv(void*, int, MPI_Datatype, void*, int *, int *, MPI_Datatype, int, MPI_Comm);
	int MPIAPI PMPI_Scatter(void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm);
	int MPIAPI PMPI_Scatterv(void*, int *, int *displs, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm);
	int MPIAPI PMPI_Allgather(void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm);
	int MPIAPI PMPI_Allgatherv(void*, int, MPI_Datatype, void*, int *, int *, MPI_Datatype, MPI_Comm);
	int MPIAPI PMPI_Alltoall(void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm);
	int MPIAPI PMPI_Alltoallv(void*, int *, int *, MPI_Datatype, void*, int *, int *, MPI_Datatype, MPI_Comm);
	int MPIAPI PMPI_Reduce(void*, void*, int, MPI_Datatype, MPI_Op, int, MPI_Comm);
	int MPIAPI PMPI_Op_create(MPI_User_function *, int, MPI_Op *);
	int MPIAPI PMPI_Op_free(MPI_Op *);
	int MPIAPI PMPI_Allreduce(void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
	int MPIAPI PMPI_Reduce_scatter(void*, void*, int *, MPI_Datatype, MPI_Op, MPI_Comm);
	int MPIAPI PMPI_Scan(void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
	int MPIAPI PMPI_Group_size(MPI_Group, int *);
	int MPIAPI PMPI_Group_rank(MPI_Group, int *);
	int MPIAPI PMPI_Group_translate_ranks(MPI_Group, int, int *, MPI_Group, int *);
	int MPIAPI PMPI_Group_compare(MPI_Group, MPI_Group, int *);
	int MPIAPI PMPI_Comm_group(MPI_Comm, MPI_Group *);
	int MPIAPI PMPI_Group_union(MPI_Group, MPI_Group, MPI_Group *);
	int MPIAPI PMPI_Group_intersection(MPI_Group, MPI_Group, MPI_Group *);
	int MPIAPI PMPI_Group_difference(MPI_Group, MPI_Group, MPI_Group *);
	int MPIAPI PMPI_Group_incl(MPI_Group, int, int *, MPI_Group *);
	int MPIAPI PMPI_Group_excl(MPI_Group, int, int *, MPI_Group *);
	int MPIAPI PMPI_Group_range_incl(MPI_Group, int, int[][3], MPI_Group *);
	int MPIAPI PMPI_Group_range_excl(MPI_Group, int, int[][3], MPI_Group *);
	int MPIAPI PMPI_Group_free(MPI_Group *);
	int MPIAPI PMPI_Comm_size(MPI_Comm, int *);
	int MPIAPI PMPI_Comm_rank(MPI_Comm, int *);
	int MPIAPI PMPI_Comm_compare(MPI_Comm, MPI_Comm, int *);
	int MPIAPI PMPI_Comm_dup(MPI_Comm, MPI_Comm *);
	int MPIAPI PMPI_Comm_create(MPI_Comm, MPI_Group, MPI_Comm *);
	int MPIAPI PMPI_Comm_split(MPI_Comm, int, int, MPI_Comm *);
	int MPIAPI PMPI_Comm_free(MPI_Comm *);
	int MPIAPI PMPI_Comm_test_inter(MPI_Comm, int *);
	int MPIAPI PMPI_Comm_remote_size(MPI_Comm, int *);
	int MPIAPI PMPI_Comm_remote_group(MPI_Comm, MPI_Group *);
	int MPIAPI PMPI_Intercomm_create(MPI_Comm, int, MPI_Comm, int, int, MPI_Comm *);
	int MPIAPI PMPI_Intercomm_merge(MPI_Comm, int, MPI_Comm *);
	int MPIAPI PMPI_Keyval_create(MPI_Copy_function *, MPI_Delete_function *, int *, void*);
	int MPIAPI PMPI_Keyval_free(int *);
	int MPIAPI PMPI_Attr_put(MPI_Comm, int, void*);
	int MPIAPI PMPI_Attr_get(MPI_Comm, int, void *, int *);
	int MPIAPI PMPI_Attr_delete(MPI_Comm, int);
	int MPIAPI PMPI_Topo_test(MPI_Comm, int *);
	int MPIAPI PMPI_Cart_create(MPI_Comm, int, int *, int *, int, MPI_Comm *);
	int MPIAPI PMPI_Dims_create(int, int, int *);
	int MPIAPI PMPI_Graph_create(MPI_Comm, int, int *, int *, int, MPI_Comm *);
	int MPIAPI PMPI_Graphdims_get(MPI_Comm, int *, int *);
	int MPIAPI PMPI_Graph_get(MPI_Comm, int, int, int *, int *);
	int MPIAPI PMPI_Cartdim_get(MPI_Comm, int *);
	int MPIAPI PMPI_Cart_get(MPI_Comm, int, int *, int *, int *);
	int MPIAPI PMPI_Cart_rank(MPI_Comm, int *, int *);
	int MPIAPI PMPI_Cart_coords(MPI_Comm, int, int, int *);
	int MPIAPI PMPI_Graph_neighbors_count(MPI_Comm, int, int *);
	int MPIAPI PMPI_Graph_neighbors(MPI_Comm, int, int, int *);
	int MPIAPI PMPI_Cart_shift(MPI_Comm, int, int, int *, int *);
	int MPIAPI PMPI_Cart_sub(MPI_Comm, int *, MPI_Comm *);
	int MPIAPI PMPI_Cart_map(MPI_Comm, int, int *, int *, int *);
	int MPIAPI PMPI_Graph_map(MPI_Comm, int, int *, int *, int *);
	int MPIAPI PMPI_Get_processor_name(char *, int *);
	int MPIAPI PMPI_Get_version(int *, int *);
	int MPIAPI PMPI_Errhandler_create(MPI_Handler_function *, MPI_Errhandler *);
	int MPIAPI PMPI_Errhandler_set(MPI_Comm, MPI_Errhandler);
	int MPIAPI PMPI_Errhandler_get(MPI_Comm, MPI_Errhandler *);
	int MPIAPI PMPI_Errhandler_free(MPI_Errhandler *);
	int MPIAPI PMPI_Error_string(int, char *, int *);
	int MPIAPI PMPI_Error_class(int, int *);

	int MPIAPI PMPI_Type_get_envelope(MPI_Datatype, int *, int *, int *, int *);

	/* Wtime done above */
	int MPIAPI PMPI_Init(int *, char ***);
	int MPIAPI PMPI_Finalize(void);
	int MPIAPI PMPI_Initialized(int *);
	int MPIAPI PMPI_Abort(MPI_Comm, int);

	int MPIAPI PMPI_Pcontrol(const int, ...);

	/* MPI-2 functions */

	/* Process Creation and Management */
	int MPIAPI PMPI_Close_port(char *);
	int MPIAPI PMPI_Comm_accept(char *, MPI_Info, int, MPI_Comm, MPI_Comm *);
	int MPIAPI PMPI_Comm_connect(char *, MPI_Info, int, MPI_Comm, MPI_Comm *);
	int MPIAPI PMPI_Comm_disconnect(MPI_Comm *);
	int MPIAPI PMPI_Comm_get_parent(MPI_Comm *);
	int MPIAPI PMPI_Comm_join(int, MPI_Comm *);
	int MPIAPI PMPI_Comm_spawn(char *, char *[], int, MPI_Info, int, MPI_Comm, MPI_Comm *,
		int[]);
	int MPIAPI PMPI_Comm_spawn_multiple(int, char *[], char **[], int[], MPI_Info[], int,
		MPI_Comm, MPI_Comm *, int[]);
	int MPIAPI PMPI_Lookup_name(char *, MPI_Info, char *);
	int MPIAPI PMPI_Open_port(MPI_Info, char *);
	int MPIAPI PMPI_Publish_name(char *, MPI_Info, char *);
	int MPIAPI PMPI_Unpublish_name(char *, MPI_Info, char *);

	/* One-Sided Communications */
	int MPIAPI PMPI_Accumulate(void *, int, MPI_Datatype, int, MPI_Aint, int,
		MPI_Datatype, MPI_Op, MPI_Win);
	int MPIAPI PMPI_Get(void *, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype,
		MPI_Win);
	int MPIAPI PMPI_Put(void *, int, MPI_Datatype, int, MPI_Aint, int, MPI_Datatype,
		MPI_Win);
	int MPIAPI PMPI_Win_complete(MPI_Win);
	int MPIAPI PMPI_Win_create(void *, MPI_Aint, int, MPI_Info, MPI_Comm, MPI_Win *);
	int MPIAPI PMPI_Win_fence(int, MPI_Win);
	int MPIAPI PMPI_Win_free(MPI_Win *);
	int MPIAPI PMPI_Win_get_group(MPI_Win, MPI_Group *);
	int MPIAPI PMPI_Win_lock(int, int, int, MPI_Win);
	int MPIAPI PMPI_Win_post(MPI_Group, int, MPI_Win);
	int MPIAPI PMPI_Win_start(MPI_Group, int, MPI_Win);
	int MPIAPI PMPI_Win_test(MPI_Win, int *);
	int MPIAPI PMPI_Win_unlock(int, MPI_Win);
	int MPIAPI PMPI_Win_wait(MPI_Win);

	/* Extended Collective Operations */
	int MPIAPI PMPI_Alltoallw(void *, int[], int[], MPI_Datatype[], void *, int[],
		int[], MPI_Datatype[], MPI_Comm);
	int MPIAPI PMPI_Exscan(void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);

	/* External Interfaces */
	int MPIAPI PMPI_Add_error_class(int *);
	int MPIAPI PMPI_Add_error_code(int, int *);
	int MPIAPI PMPI_Add_error_string(int, char *);
	int MPIAPI PMPI_Comm_call_errhandler(MPI_Comm, int);
	int MPIAPI PMPI_Comm_create_keyval(MPI_Comm_copy_attr_function *,
		MPI_Comm_delete_attr_function *, int *, void *);
	int MPIAPI PMPI_Comm_delete_attr(MPI_Comm, int);
	int MPIAPI PMPI_Comm_free_keyval(int *);
	int MPIAPI PMPI_Comm_get_attr(MPI_Comm, int, void *, int *);
	int MPIAPI PMPI_Comm_get_name(MPI_Comm, char *, int *);
	int MPIAPI PMPI_Comm_set_attr(MPI_Comm, int, void *);
	int MPIAPI PMPI_Comm_set_name(MPI_Comm, char *);
	int MPIAPI PMPI_File_call_errhandler(MPI_File, int);
	int MPIAPI PMPI_Grequest_complete(MPI_Request);
	int MPIAPI PMPI_Grequest_start(MPI_Grequest_query_function *,
		MPI_Grequest_free_function *,
		MPI_Grequest_cancel_function *, void *, MPI_Request *);
	int MPIAPI PMPI_Init_thread(int *, char ***, int, int *);
	int MPIAPI PMPI_Is_thread_main(int *);
	int MPIAPI PMPI_Query_thread(int *);
	int MPIAPI PMPI_Status_set_cancelled(MPI_Status *, int);
	int MPIAPI PMPI_Status_set_elements(MPI_Status *, MPI_Datatype, int);
	int MPIAPI PMPI_Type_create_keyval(MPI_Type_copy_attr_function *,
		MPI_Type_delete_attr_function *, int *, void *);
	int MPIAPI PMPI_Type_delete_attr(MPI_Datatype, int);
	int MPIAPI PMPI_Type_dup(MPI_Datatype, MPI_Datatype *);
	int MPIAPI PMPI_Type_free_keyval(int *);
	int MPIAPI PMPI_Type_get_attr(MPI_Datatype, int, void *, int *);
	int MPIAPI PMPI_Type_get_contents(MPI_Datatype, int, int, int, int[], MPI_Aint[],
		MPI_Datatype[]);
	int MPIAPI PMPI_Type_get_envelope(MPI_Datatype, int *, int *, int *, int *);
	int MPIAPI PMPI_Type_get_name(MPI_Datatype, char *, int *);
	int MPIAPI PMPI_Type_set_attr(MPI_Datatype, int, void *);
	int MPIAPI PMPI_Type_set_name(MPI_Datatype, char *);
	int MPIAPI PMPI_Type_match_size(int, int, MPI_Datatype *);
	int MPIAPI PMPI_Win_call_errhandler(MPI_Win, int);
	int MPIAPI PMPI_Win_create_keyval(MPI_Win_copy_attr_function *,
		MPI_Win_delete_attr_function *, int *, void *);
	int MPIAPI PMPI_Win_delete_attr(MPI_Win, int);
	int MPIAPI PMPI_Win_free_keyval(int *);
	int MPIAPI PMPI_Win_get_attr(MPI_Win, int, void *, int *);
	int MPIAPI PMPI_Win_get_name(MPI_Win, char *, int *);
	int MPIAPI PMPI_Win_set_attr(MPI_Win, int, void *);
	int MPIAPI PMPI_Win_set_name(MPI_Win, char *);

	/* Fortran 90-related functions.  These routines are available only if
	Fortran 90 support is enabled
	*/
	int MPIAPI PMPI_Type_create_f90_integer(int, MPI_Datatype *);
	int MPIAPI PMPI_Type_create_f90_real(int, int, MPI_Datatype *);
	int MPIAPI PMPI_Type_create_f90_complex(int, int, MPI_Datatype *);

	/* Miscellany */
	int MPIAPI PMPI_Alloc_mem(MPI_Aint, MPI_Info info, void *baseptr);
	int MPIAPI PMPI_Comm_create_errhandler(MPI_Comm_errhandler_fn *, MPI_Errhandler *);
	int MPIAPI PMPI_Comm_get_errhandler(MPI_Comm, MPI_Errhandler *);
	int MPIAPI PMPI_Comm_set_errhandler(MPI_Comm, MPI_Errhandler);
	int MPIAPI PMPI_File_create_errhandler(MPI_File_errhandler_fn *, MPI_Errhandler *);
	int MPIAPI PMPI_File_get_errhandler(MPI_File, MPI_Errhandler *);
	int MPIAPI PMPI_File_set_errhandler(MPI_File, MPI_Errhandler);
	int MPIAPI PMPI_Finalized(int *);
	int MPIAPI PMPI_Free_mem(void *);
	int MPIAPI PMPI_Get_address(void *, MPI_Aint *);
	int MPIAPI PMPI_Info_create(MPI_Info *);
	int MPIAPI PMPI_Info_delete(MPI_Info, char *);
	int MPIAPI PMPI_Info_dup(MPI_Info, MPI_Info *);
	int MPIAPI PMPI_Info_free(MPI_Info *info);
	int MPIAPI PMPI_Info_get(MPI_Info, char *, int, char *, int *);
	int MPIAPI PMPI_Info_get_nkeys(MPI_Info, int *);
	int MPIAPI PMPI_Info_get_nthkey(MPI_Info, int, char *);
	int MPIAPI PMPI_Info_get_valuelen(MPI_Info, char *, int *, int *);
	int MPIAPI PMPI_Info_set(MPI_Info, char *, char *);
	int MPIAPI PMPI_Pack_external(char *, void *, int, MPI_Datatype, void *, MPI_Aint,
		MPI_Aint *);
	int MPIAPI PMPI_Pack_external_size(char *, int, MPI_Datatype, MPI_Aint *);
	int MPIAPI PMPI_Request_get_status(MPI_Request, int *, MPI_Status *);
	int MPIAPI PMPI_Status_c2f(MPI_Status *, MPI_Fint *);
	int MPIAPI PMPI_Status_f2c(MPI_Fint *, MPI_Status *);
	int MPIAPI PMPI_Type_create_darray(int, int, int, int[], int[], int[], int[], int,
		MPI_Datatype, MPI_Datatype *);
	int MPIAPI PMPI_Type_create_hindexed(int, int[], MPI_Aint[], MPI_Datatype,
		MPI_Datatype *);
	int MPIAPI PMPI_Type_create_hvector(int, int, MPI_Aint, MPI_Datatype, MPI_Datatype *);
	int MPIAPI PMPI_Type_create_indexed_block(int, int, int[], MPI_Datatype,
		MPI_Datatype *);
	int MPIAPI PMPI_Type_create_resized(MPI_Datatype, MPI_Aint, MPI_Aint, MPI_Datatype *);
	int MPIAPI PMPI_Type_create_struct(int, int[], MPI_Aint[], MPI_Datatype[],
		MPI_Datatype *);
	int MPIAPI PMPI_Type_create_subarray(int, int[], int[], int[], int, MPI_Datatype,
		MPI_Datatype *);
	int MPIAPI PMPI_Type_get_extent(MPI_Datatype, MPI_Aint *, MPI_Aint *);
	int MPIAPI PMPI_Type_get_true_extent(MPI_Datatype, MPI_Aint *, MPI_Aint *);
	int MPIAPI PMPI_Unpack_external(char *, void *, MPI_Aint, MPI_Aint *, void *, int,
		MPI_Datatype);
	int MPIAPI PMPI_Win_create_errhandler(MPI_Win_errhandler_fn *, MPI_Errhandler *);
	int MPIAPI PMPI_Win_get_errhandler(MPI_Win, MPI_Errhandler *);
	int MPIAPI PMPI_Win_set_errhandler(MPI_Win, MPI_Errhandler);
#endif  /* MPI_BUILD_PROFILING */
	/* End of MPI bindings */

	/* feature advertisement */
#define MPIIMPL_ADVERTISES_FEATURES 1
#define MPIIMPL_HAVE_MPI_INFO 1                                                 
#define MPIIMPL_HAVE_MPI_COMBINER_DARRAY 1                                      
#define MPIIMPL_HAVE_MPI_TYPE_CREATE_DARRAY 1
#define MPIIMPL_HAVE_MPI_COMBINER_SUBARRAY 1                                    
#define MPIIMPL_HAVE_MPI_TYPE_CREATE_DARRAY 1
#define MPIIMPL_HAVE_MPI_COMBINER_DUP 1                                         
#define MPIIMPL_HAVE_MPI_GREQUEST 1      
#define MPIIMPL_HAVE_STATUS_SET_BYTES 1
#define MPIIMPL_HAVE_STATUS_SET_INFO 1

#include "mpio.h"

#if defined(__cplusplus)
}
#endif

#endif /* MPI_INCLUDED */
