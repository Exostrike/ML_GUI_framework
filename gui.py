# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Jun 28 2017)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
import pandas as pd
from overall_mode import start_models, start_custom

###########################################################################
## Class setupbox
###########################################################################

class setupbox ( wx.Frame ):
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Binary Classification Control Panel", pos = wx.DefaultPosition, size = wx.Size( 551,650 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
		self.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_ACTIVEBORDER ) )
		
		bSizer2 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText3 = wx.StaticText( self, wx.ID_ANY, u"Data file", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3.Wrap( -1 )
		bSizer2.Add( self.m_staticText3, 0, wx.ALL, 5 )
		
		self.fileselect = wx.FilePickerCtrl( self, wx.ID_ANY, wx.EmptyString, u"Select a file", u"*.*", wx.DefaultPosition, wx.DefaultSize, wx.FLP_DEFAULT_STYLE )
		self.fileselect.SetToolTip( u"Select file to be analysised" )
		
		bSizer2.Add( self.fileselect, 0, wx.ALL, 5 )
		
		self.m_staticText5 = wx.StaticText( self, wx.ID_ANY, u"Target variable:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText5.Wrap( -1 )
		bSizer2.Add( self.m_staticText5, 0, wx.ALL, 5 )
		
		m_comboBox1Choices = []
		self.m_comboBox1 = wx.ComboBox( self, wx.ID_ANY, u"Select variable", wx.DefaultPosition, wx.DefaultSize, m_comboBox1Choices, 0 )
		bSizer2.Add( self.m_comboBox1, 0, wx.ALL, 5 )
		
		gSizer41 = wx.GridSizer( 1, 4, 0, 0 )
		
		self.m_staticText71 = wx.StaticText( self, wx.ID_ANY, u"secondary variable:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText71.Wrap( -1 )
		gSizer41.Add( self.m_staticText71, 0, wx.ALL, 5 )
		
		m_comboBox2Choices = []
		self.m_comboBox2 = wx.ComboBox( self, wx.ID_ANY, u"Select variable", wx.DefaultPosition, wx.DefaultSize, m_comboBox2Choices, 0 )
		gSizer41.Add( self.m_comboBox2, 0, wx.ALL, 5 )
		
		self.m_staticText8 = wx.StaticText( self, wx.ID_ANY, u"Greater than value:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText8.Wrap( -1 )
		gSizer41.Add( self.m_staticText8, 0, wx.ALL, 5 )
		
		self.secondary_value = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer41.Add( self.secondary_value, 0, wx.ALL, 5 )
		
		
		bSizer2.Add( gSizer41, 0, 0, 5 )
		
		gSizer1 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText51 = wx.StaticText( self, wx.ID_ANY, u"remove target values below:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText51.Wrap( -1 )
		gSizer1.Add( self.m_staticText51, 0, wx.ALL, 5 )
		
		self.removebox = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.removebox.SetToolTip( u"remove unwanted observations" )
		
		gSizer1.Add( self.removebox, 0, wx.ALL, 5 )
		
		self.m_staticText7 = wx.StaticText( self, wx.ID_ANY, u"Classification boundary:", wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_LEFT|wx.ALIGN_RIGHT )
		self.m_staticText7.Wrap( -1 )
		gSizer1.Add( self.m_staticText7, 0, wx.ALL, 5 )
		
		self.boundarybox = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.boundarybox.SetToolTip( u"Input the boundary for the binary classification" )
		
		gSizer1.Add( self.boundarybox, 0, wx.ALL, 5 )
		
		
		bSizer2.Add( gSizer1, 0, wx.ALL, 5 )
		
		gSizer4 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText6 = wx.StaticText( self, wx.ID_ANY, u"Drop variables:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText6.Wrap( -1 )
		gSizer4.Add( self.m_staticText6, 0, wx.ALL, 5 )
		
		m_comboBox3Choices = []
		self.m_comboBox3 = wx.ComboBox( self, wx.ID_ANY, u"Select variable", wx.DefaultPosition, wx.DefaultSize, m_comboBox3Choices, 0 )
		gSizer4.Add( self.m_comboBox3, 0, wx.ALL, 5 )
		
		self.dropvaribles = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE )
		self.dropvaribles.SetToolTip( u"Enter variables you want to drop from the dataset" )
		
		gSizer4.Add( self.dropvaribles, 0, wx.ALL, 5 )
		
		
		bSizer2.Add( gSizer4, 0, 0, 5 )
		
		gSizer3 = wx.GridSizer( 0, 3, 0, 0 )
		
		self.m_staticText81 = wx.StaticText( self, wx.ID_ANY, u"Base Models", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText81.Wrap( -1 )
		self.m_staticText81.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString ) )
		
		gSizer3.Add( self.m_staticText81, 0, wx.ALL, 5 )
		
		
		gSizer3.AddSpacer(10)
		
		
		gSizer3.AddSpacer(10)
		
		self.basetick1 = wx.CheckBox( self, wx.ID_ANY, u"Decision trees", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.basetick1.SetToolTip( u"Select models to run" )
		
		gSizer3.Add( self.basetick1, 0, wx.ALL, 5 )
		
		self.basetick2 = wx.CheckBox( self, wx.ID_ANY, u"k nearest neighbour", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.basetick2.SetToolTip( u"Select models to run" )
		
		gSizer3.Add( self.basetick2, 0, wx.ALL, 5 )
		
		self.basetick3 = wx.CheckBox( self, wx.ID_ANY, u"Other models", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.basetick3, 0, wx.ALL, 5 )
		
		self.m_staticText811 = wx.StaticText( self, wx.ID_ANY, u"Hetrogeneous Models", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText811.Wrap( -1 )
		self.m_staticText811.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString ) )
		
		gSizer3.Add( self.m_staticText811, 0, wx.ALL, 5 )
		
		
		gSizer3.AddSpacer(10 )
		
		
		gSizer3.AddSpacer(10 )
		
		self.hetrotick1 = wx.CheckBox( self, wx.ID_ANY, u"voting models", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.hetrotick1.SetToolTip( u"Select models to run" )
		
		gSizer3.Add( self.hetrotick1, 0, wx.ALL, 5 )
		
		self.hetrotick2 = wx.CheckBox( self, wx.ID_ANY, u"stacking models", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.hetrotick2.SetToolTip( u"Select models to run" )
		
		gSizer3.Add( self.hetrotick2, 0, wx.ALL, 5 )
		
		
		gSizer3.AddSpacer(10 )
		
		self.m_staticText8111 = wx.StaticText( self, wx.ID_ANY, u"Homogeneous Model", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText8111.Wrap( -1 )
		self.m_staticText8111.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString ) )
		
		gSizer3.Add( self.m_staticText8111, 0, wx.ALL, 5 )
		
		
		gSizer3.AddSpacer( 10)
		
		
		gSizer3.AddSpacer(10 )
		
		self.homotick1 = wx.CheckBox( self, wx.ID_ANY, u"Boosting models", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.homotick1, 0, wx.ALL, 5 )
		
		self.homotick2 = wx.CheckBox( self, wx.ID_ANY, u"Bagging models", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.homotick2, 0, wx.ALL, 5 )
		
		self.homotick3 = wx.CheckBox( self, wx.ID_ANY, u"Other models", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.homotick3, 0, wx.ALL, 5 )
		
		self.m_staticText81111 = wx.StaticText( self, wx.ID_ANY, u"Hybrid Models", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText81111.Wrap( -1 )
		self.m_staticText81111.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString ) )
		
		gSizer3.Add( self.m_staticText81111, 0, wx.ALL, 5 )
		
		
		gSizer3.AddSpacer(10)
		
		
		gSizer3.AddSpacer(10)
		
		self.hybridtick1 = wx.CheckBox( self, wx.ID_ANY, u"hybrid using hetrogenous", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.hybridtick1, 0, wx.ALL, 5 )
		
		self.hybridtick2 = wx.CheckBox( self, wx.ID_ANY, u"hybrid using homogenous", wx.DefaultPosition, wx.DefaultSize, 0 )
		gSizer3.Add( self.hybridtick2, 0, wx.ALL, 5 )
		
		
		gSizer3.AddSpacer(10)
		
		
		bSizer2.Add( gSizer3, 0, 0, 5 )
		
		self.custom_model = wx.Button( self, wx.ID_ANY, u"Custom Model", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer2.Add( self.custom_model, 0, wx.ALL, 5 )
		
		self.m_button2 = wx.Button( self, wx.ID_ANY, u"Activate", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer2.Add( self.m_button2, 0, wx.ALL, 5 )
		
		
		self.SetSizer( bSizer2 )
		self.Layout()
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.fileselect.Bind( wx.EVT_FILEPICKER_CHANGED, self.selectfile )
		self.m_comboBox3.Bind( wx.EVT_COMBOBOX, self.dropselect )
		self.custom_model.Bind( wx.EVT_BUTTON, self.custommodel)
		self.m_button2.Bind( wx.EVT_BUTTON, self.activation )
	
	def __del__( self ):
		pass

	
	
	# Virtual event handlers, overide them in your derived class
	def selectfile( self, event ):
		file_location = self.fileselect.GetPath()
		data = pd.read_csv(file_location, nrows=0)
		data = data.columns.tolist()
		self.m_comboBox1.Append(data)
		self.m_comboBox2.Append(data)
		self.m_comboBox3.Append(data)         

	def custommodel( self, event ):
		self.secondWindow = MyFrame2(self)
		self.secondWindow.Show()       
		
	def dropselect( self, event ):
		removed_class = str(self.m_comboBox3.GetValue())
		self.dropvaribles.WriteText(removed_class+str('\n'))		
		
	def activation( self, event ):
		file_location = self.fileselect.GetPath()
		target_class = self.m_comboBox1.GetValue()
		removed_boundary = int(self.removebox.GetValue())
		class_boundary = int(self.boundarybox.GetValue())
		secondary_class = self.m_comboBox2.GetValue()
		secondary_boundary = int(self.secondary_value.GetValue())
		dropped = self.dropvaribles.GetValue().split()
		base_check = (self.basetick1.IsChecked()|self.basetick2.IsChecked()|self.basetick3.IsChecked())
		base1 = (self.basetick1.IsChecked())
		base2 = (self.basetick2.IsChecked())
		base3 = (self.basetick3.IsChecked())
		hetro_check = (self.hetrotick1.IsChecked()|self.hetrotick2.IsChecked())
		hetro1 = (self.hetrotick1.IsChecked())
		hetro2 = (self.hetrotick2.IsChecked())  
		homo_check = (self.homotick1.IsChecked()|self.homotick2.IsChecked()|self.homotick3.IsChecked())
		homo1 = (self.homotick1.IsChecked())
		homo2 = (self.homotick2.IsChecked())
		homo3 = (self.homotick3.IsChecked())
		hybrid_check = (self.hybridtick1.IsChecked()|self.hybridtick2.IsChecked())
		hybrid1 = (self.hybridtick1.IsChecked())
		hybrid2 = (self.hybridtick2.IsChecked())
		start_models(file_location,target_class,removed_boundary,class_boundary,secondary_class,secondary_boundary,dropped,base_check,base1,base2,base3,hetro_check,hetro1,hetro2,homo_check,homo1,homo2,homo3,hybrid_check,hybrid1,hybrid2)

class MyFrame2 ( wx.Frame ):
	
	def __init__( self, parent):
		wx.Frame.__init__ ( self, None, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 450,344 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		self.parent = parent
		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
		
		bSizer2 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText12 = wx.StaticText( self, wx.ID_ANY, u"Custom model creation", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText12.Wrap( -1 )
		self.m_staticText12.SetFont( wx.Font( wx.NORMAL_FONT.GetPointSize(), wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, wx.EmptyString ) )
		
		bSizer2.Add( self.m_staticText12, 0, wx.ALL, 5 )
		
		self.m_staticline4 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer2.Add( self.m_staticline4, 0, wx.EXPAND |wx.ALL, 5 )
		
		self.m_radioBtn2 = wx.RadioButton( self, wx.ID_ANY, u"Hetrogeneous", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer2.Add( self.m_radioBtn2, 0, wx.ALL, 5 )
		
		m_comboBox4Choices = [ u"Voting", u"Stacking" ]
		self.m_comboBox4 = wx.ComboBox( self, wx.ID_ANY, u"method", wx.DefaultPosition, wx.DefaultSize, m_comboBox4Choices, 0 )
		bSizer2.Add( self.m_comboBox4, 0, wx.ALL, 5 )
		
		gSizer4 = wx.GridSizer( 0, 2, 0, 0 )
		
		self.m_staticText6 = wx.StaticText( self, wx.ID_ANY, u"Drop variables:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText6.Wrap( -1 )
		gSizer4.Add( self.m_staticText6, 0, wx.ALL, 5 )
		
		m_comboBox3Choices = [u"Decision_Tree",u"K_nearest_neighbor",u"GaussianNB",u"MLPClassifier",u"BernoulliNB",u"Extra_Tree",u"Linear_Discriminant_Analysis",u"Quadratic_Discriminant_Analysis"]
		self.m_comboBox3 = wx.ComboBox( self, wx.ID_ANY, u"Select variable", wx.DefaultPosition, wx.DefaultSize, m_comboBox3Choices, 0 )
		gSizer4.Add( self.m_comboBox3, 0, wx.ALL, 5 )
		
		self.dropvaribles = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE )
		self.dropvaribles.SetToolTip( u"Enter variables you want to drop from the dataset" )
		
		gSizer4.Add( self.dropvaribles, 0, wx.ALL, 5 )
		
		
		bSizer2.Add( gSizer4, 0, 0, 5 )
		
		self.m_staticline3 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
		bSizer2.Add( self.m_staticline3, 0, wx.EXPAND |wx.ALL, 5 )
		
		self.m_radioBtn3 = wx.RadioButton( self, wx.ID_ANY, u"Homogeneous", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer2.Add( self.m_radioBtn3, 0, wx.ALL, 5 )
		
		gSizer8 = wx.GridSizer( 0, 2, 0, 0 )
		
		m_comboBox7Choices = [u"AdaBoost",u"Bagging",u"Random_Forest",u"Gradient_Boosting"]
		self.m_comboBox7 = wx.ComboBox( self, wx.ID_ANY, u"Method", wx.DefaultPosition, wx.DefaultSize, m_comboBox7Choices, 0 )
		gSizer8.Add( self.m_comboBox7, 0, wx.ALL, 5 )
		
		m_comboBox8Choices = [u"Decision_Tree",u"K_nearest_neighbor",u"GaussianNB",u"MLPClassifier",u"BernoulliNB",u"Extra_Tree",u"Linear_Discriminant_Analysis",u"Quadratic_Discriminant_Analysis"]
		self.m_comboBox8 = wx.ComboBox( self, wx.ID_ANY, u"Model", wx.DefaultPosition, wx.DefaultSize, m_comboBox8Choices, 0 )
		gSizer8.Add( self.m_comboBox8, 0, wx.ALL, 5 )
		
		
		bSizer2.Add( gSizer8, 0, 0, 5 )
		
		self.m_button3 = wx.Button( self, wx.ID_ANY, u"Create model", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer2.Add( self.m_button3, 0, wx.ALL, 5 )
		
		
		self.SetSizer( bSizer2 )
		self.Layout()
		
		self.Centre( wx.BOTH )
		
		# Connect Events
		self.m_comboBox3.Bind( wx.EVT_COMBOBOX, self.addselect )
		self.m_button3.Bind( wx.EVT_BUTTON, self.activatecustom )
	
	def __del__( self ):
		pass
	
	
	# Virtual event handlers, overide them in your derived class
	def addselect( self, event ):
		removed_class = str(self.m_comboBox3.GetValue())
		self.dropvaribles.WriteText(removed_class+str('\n'))	
	
	def activatecustom( self, event ):
		if self.m_radioBtn2.GetValue()== True:
				hetromodels = self.dropvaribles.GetValue().split()
				hetromethod = self.m_comboBox4.GetValue()
				file_location = self.parent.fileselect.GetPath()
				target_class = self.parent.m_comboBox1.GetValue()
				removed_boundary = int(self.parent.removebox.GetValue())
				class_boundary = int(self.parent.boundarybox.GetValue())
				secondary_class = self.parent.m_comboBox2.GetValue()
				secondary_boundary = int(self.parent.secondary_value.GetValue())
				dropped = self.parent.dropvaribles.GetValue().split()
				start_custom(1,file_location,target_class,removed_boundary,class_boundary,secondary_class,secondary_boundary,dropped,hetromethod,hetromodels)
                
                
		else :
				homomethod = self.m_comboBox7.GetValue()
				homomodel = self.m_comboBox8.GetValue()
				file_location = self.parent.fileselect.GetPath()
				target_class = self.parent.m_comboBox1.GetValue()
				removed_boundary = int(self.parent.removebox.GetValue())
				class_boundary = int(self.parent.boundarybox.GetValue())
				secondary_class = self.parent.m_comboBox2.GetValue()
				secondary_boundary = int(self.parent.secondary_value.GetValue())
				dropped = self.parent.dropvaribles.GetValue().split()
				start_custom(2,file_location,target_class,removed_boundary,class_boundary,secondary_class,secondary_boundary,dropped,homomethod,homomodel)                   

app = wx.App(False)
 

frame = setupbox(None)
frame.Show(True)
app.MainLoop()
