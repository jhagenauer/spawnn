package spawnn.gui;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;

public class NGPanel extends JPanel {

	private static final long serialVersionUID = -1010659930429575440L;
	
	private JComboBox nbhBox, adaptBox;
	enum decayFunctions {Linear,Power}
	
	private JTextField textField_3;
	private JTextField textField_4;
	private JTextField textField_5;
	private JTextField textField_6;
	private JTextField textField_7;

	public NGPanel() {
		setLayout(new MigLayout());
		
		JLabel lblNeurons = new JLabel("Neurons:");
		add(lblNeurons, "");
		
		textField_3 = new JTextField();
		textField_3.setText("25");
		textField_3.setColumns(10);
		add(textField_3, "wrap");
		
		JLabel lblNbh = new JLabel("Neighborhood:" );
		add( lblNbh, "");
		nbhBox = new JComboBox(new DefaultComboBoxModel(decayFunctions.values()));
		nbhBox.setSelectedIndex(1);
		add(nbhBox, "");
		
		JLabel lblLambdaInit = new JLabel("Init:");
		add(lblLambdaInit, "");
		
		textField_4 = new JTextField();
		textField_4.setText("25");
		textField_4.setColumns(10);
		add(textField_4, "");
		
		JLabel lblLambdaFinal = new JLabel("Final:");
		add(lblLambdaFinal, "");
		
		textField_5 = new JTextField();
		textField_5.setText("0.01");
		textField_5.setColumns(10);
		add(textField_5, "wrap");
		
		JLabel lblAdapt = new JLabel("Adaptation:" );
		add( lblAdapt, "");
		adaptBox = new JComboBox(new DefaultComboBoxModel(decayFunctions.values()));
		adaptBox.setSelectedIndex(1);
		add(adaptBox, "");
		
		JLabel lblEpsilonInit = new JLabel("Init:");
		add(lblEpsilonInit, "");
		
		textField_6 = new JTextField();
		textField_6.setText("0.5");
		add(textField_6, "");
		textField_6.setColumns(10);
		
		JLabel lblEpsilonFinal = new JLabel("Final:");
		add(lblEpsilonFinal, "");
		
		textField_7 = new JTextField();
		textField_7.setText("0.01");
		add(textField_7, "");
		textField_7.setColumns(10);
	}
	
	public DecayFunction getNeighborhoodRate() {
		double init = Double.parseDouble(textField_4.getText());
		double fin = Double.parseDouble(textField_5.getText());
		
		if( nbhBox.getSelectedItem() == decayFunctions.Linear )
			return new LinearDecay(init, fin);
		else
			return new PowerDecay(init, fin);
	}
	
	public DecayFunction getAdaptationRate() {
		double init = Double.parseDouble(textField_6.getText());
		double fin = Double.parseDouble(textField_7.getText());
		
		if( adaptBox.getSelectedItem() == decayFunctions.Linear )
			return new LinearDecay(init, fin);
		else
			return new PowerDecay(init, fin);
	}
	
	public int numNeurons() {
		return Integer.parseInt(textField_3.getText());
	}

}
