from loaddata import loaddata
from base_models3 import base, hetro, homo, hybrid,custom_hetro,custom_homo

def start_models(file_location,target_class,removed_boundary,class_boundary,secondary_class,secondary_boundary,dropped,base_check,base1,base2,base3,hetro_check,hetro1,hetro2,homo_check,homo1,homo2,homo3,hybrid_check,hybrid1,hybrid2):

	X_train, X_test, y_train, y_test,n_classes = loaddata(file_location,target_class,removed_boundary,class_boundary,secondary_class,secondary_boundary,dropped)
	
	if (base_check == True):
		base(X_train, X_test, y_train, y_test,n_classes,base1,base2,base3)
	if (hetro_check == True):	
		hetro(X_train, X_test, y_train, y_test,n_classes,hetro1,hetro2)
	if (homo_check == True):
		homo(X_train, X_test, y_train, y_test,n_classes,homo1,homo2,homo3)
	if (hybrid_check == True):	
		hybrid(X_train, X_test, y_train, y_test,n_classes,hybrid1,hybrid2)

def start_custom(status,file_location,target_class,removed_boundary,class_boundary,secondary_class,secondary_boundary,dropped,method,model):

	X_train, X_test, y_train, y_test,n_classes = loaddata(file_location,target_class,removed_boundary,class_boundary,secondary_class,secondary_boundary,dropped)
	
	if (status == 1):
		custom_hetro(X_train, X_test, y_train, y_test,n_classes,method,model)
	else:	
		custom_homo(X_train, X_test, y_train, y_test,n_classes,method,model)