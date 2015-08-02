package spawnn.gui;

import javax.swing.JPanel;
import javax.swing.JLabel;
import net.miginfocom.swing.MigLayout;
import javax.swing.JTextField;

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
