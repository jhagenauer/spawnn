package spawnn.gui;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;

public class GeoSOMPanel extends JPanel {

	private static final long serialVersionUID = -7145338010761179365L;
	private JTextField textField;

	public GeoSOMPanel() {
		setLayout(new MigLayout());
				
		JLabel lblRadius = new JLabel("Radius:");
		add(lblRadius, "");
		
		textField = new JTextField();
		textField.setText("1");
		add(textField, "growx");
		textField.setColumns(10);

	}
	
	public int getRadius() {
		int i = Integer.parseInt(textField.getText());
		return i;
	}

}
