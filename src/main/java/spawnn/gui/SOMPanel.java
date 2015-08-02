package spawnn.gui;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.BubbleKernel;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.kernel.KernelFunction;

import net.miginfocom.swing.MigLayout;

public class SOMPanel extends JPanel {
	
	private static final long serialVersionUID = -6356980461891020350L;
	private JTextField textField, textField_1, textField_2, textField_3, textField_4, textField_5;
	private JComboBox comboBox, comboBox_1, comboBox_2;
	
	enum kernel { Bubble, Gauss };
	enum type {Regular, Hexagonal};
	enum lr {Linar};

	public SOMPanel() {
		setLayout(new MigLayout());
		
		JLabel lblNewLabel = new JLabel("Grid type:");
		add(lblNewLabel, "");
		
		comboBox_1 = new JComboBox();
		comboBox_1.setModel(new DefaultComboBoxModel(type.values()));
		comboBox_1.setSelectedIndex(1);
		add(comboBox_1, "");
		
		JLabel lblXdim = new JLabel("X-Dim.:");
		add(lblXdim, "");
		
		textField = new JTextField();
		textField.setText("12");
		add(textField, "");
		textField.setColumns(10);
		
		JLabel lblYdim = new JLabel("Y-Dim.:");
		add(lblYdim, "");
		
		textField_1 = new JTextField();
		textField_1.setText("8");
		add(textField_1, "wrap");
		textField_1.setColumns(10);
		
		JLabel lblKernel = new JLabel("Kernelfunction:");
		add(lblKernel, "");
		
		comboBox = new JComboBox();
		comboBox.setModel(new DefaultComboBoxModel(kernel.values()));
		comboBox.setSelectedIndex(1);
		add(comboBox, "");
		
		JLabel lblInit = new JLabel("Init:");
		add(lblInit, "");
		
		textField_2 = new JTextField();
		textField_2.setText("10");
		add(textField_2, "");
		textField_2.setColumns(10);
		
		JLabel lblFinal = new JLabel("Final:");
		add(lblFinal, "");
		
		textField_3 = new JTextField();
		textField_3.setText("1");
		add(textField_3, "wrap");
		textField_3.setColumns(10);
		
		JLabel lblLearningrate = new JLabel("Learning rate:");
		add(lblLearningrate, "");
		
		comboBox_2 = new JComboBox();
		comboBox_2.setModel(new DefaultComboBoxModel(lr.values()));
		comboBox_2.setSelectedIndex(0);
		add(comboBox_2, "");
		
		JLabel lblInit_1 = new JLabel("Init:");
		add(lblInit_1, "");
		
		textField_4 = new JTextField();
		textField_4.setText("1.0");
		add(textField_4, "");
		textField_4.setColumns(10);
		
		JLabel lblFinal_1 = new JLabel("Final:");
		add(lblFinal_1, "");
		
		textField_5 = new JTextField();
		textField_5.setText("0.0");
		add(textField_5, "");
		textField_5.setColumns(10);
	}
	
	public DecayFunction getLearningRate() {
		double init = Double.parseDouble( textField_4.getText() );
		double fin = Double.parseDouble( textField_5.getText() );
		return new LinearDecay(init, fin);
	}
	
	public KernelFunction getKernelFunction() {
		double init = Double.parseDouble( textField_2.getText() );
		double fin = Double.parseDouble( textField_3.getText() );
		if( comboBox.getSelectedItem() == kernel.Bubble )
			return new BubbleKernel( new LinearDecay(init, fin));
		else 
			return new GaussKernel( new LinearDecay(init, fin));
	}
	
	public Grid2D<double[]> getGrid() {
		int xDim = Integer.parseInt( textField.getText() );
		int yDim = Integer.parseInt( textField_1.getText() );
		if( comboBox_1.getSelectedItem() == type.Regular )
			return new Grid2D<double[]>( xDim, yDim );
		else
			return new Grid2DHex<double[]>( xDim, yDim );	
	}
}
