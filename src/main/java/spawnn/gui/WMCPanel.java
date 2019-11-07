package spawnn.gui;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;

public class WMCPanel extends JPanel {

	private static final long serialVersionUID = 1671229973714048612L;
	private JTextField alphaField,betaField;
	
	public WMCPanel() {
		setLayout(new MigLayout(""));
		
		JLabel lblAlpha = new JLabel("Alpha:");
		add(lblAlpha, "");
		
		alphaField = new JTextField();
		alphaField.setText("0.5");
		alphaField.setColumns(10);
		add(alphaField, "");
				
		JLabel lblBeta = new JLabel("Beta:");
		add(lblBeta, "");
		
		betaField = new JTextField();
		betaField.setText("0.5");
		betaField.setColumns(10);
		add(betaField, "wrap");
	}
	
	public double getAlpha() {
		return Double.parseDouble(alphaField.getText());
	}

	public double getBeta() {
		return Double.parseDouble(betaField.getText());
	}
}
