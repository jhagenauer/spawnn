package spawnn.gui;

import javax.swing.JPanel;
import net.miginfocom.swing.MigLayout;
import javax.swing.JLabel;
import javax.swing.JTextField;

public class NGPanel extends JPanel {

	private static final long serialVersionUID = -1010659930429575440L;
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
		add(textField_3, "wrap");
		textField_3.setColumns(10);
		
		JLabel lblLambdaInit = new JLabel("Lambda init:");
		add(lblLambdaInit, "");
		
		textField_4 = new JTextField();
		textField_4.setText("25");
		add(textField_4, "");
		textField_4.setColumns(10);
		
		JLabel lblLambdaFinal = new JLabel("Lambda final:");
		add(lblLambdaFinal, "");
		
		textField_5 = new JTextField();
		textField_5.setText("0.01");
		add(textField_5, "wrap");
		textField_5.setColumns(10);
		
		JLabel lblEpsilonInit = new JLabel("Epsilon init:");
		add(lblEpsilonInit, "");
		
		textField_6 = new JTextField();
		textField_6.setText("0.5");
		add(textField_6, "");
		textField_6.setColumns(10);
		
		JLabel lblEpsilonFinal = new JLabel("Epsilon final:");
		add(lblEpsilonFinal, "");
		
		textField_7 = new JTextField();
		textField_7.setText("0.005");
		add(textField_7, "");
		textField_7.setColumns(10);
	}
	
	public double getLambdaInit() {
		return Double.parseDouble(textField_4.getText());
	}
	
	public double getLambdaFinal() {
		return Double.parseDouble(textField_5.getText());
	}
	
	public double getEpsilonInit() {
		return Double.parseDouble(textField_6.getText());
	}
	
	public double getEpsilonFinal() {
		return Double.parseDouble(textField_7.getText());
	}
	
	public int numNeurons() {
		return Integer.parseInt(textField_3.getText());
	}

}
