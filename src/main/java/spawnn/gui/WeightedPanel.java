package spawnn.gui;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;

public class WeightedPanel extends JPanel {

	private static final long serialVersionUID = -771653004644258074L;
	private JTextField textField;

	public WeightedPanel() {
		setLayout(new MigLayout());
				
		JLabel lblAlpha = new JLabel("Alpha:");
		lblAlpha.setToolTipText("Weight of spatial distance");
		add(lblAlpha, "");
		
		textField = new JTextField();
		textField.setText("0.5");
		add(textField, "growx");
		textField.setColumns(10);

	}
	
	public double getAlpha() {
		return Double.parseDouble(textField.getText());
	}

}
