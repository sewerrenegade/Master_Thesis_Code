from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import TableStyle
from reportlab.lib import colors

styles = getSampleStyleSheet()
TITLE_STYLE = styles['Title']
HEADING_STYLES = [None,styles['Heading1'],styles['Heading2'],styles['Heading3'],styles['Heading4'],styles['Heading5'],]
NORMAL_STYLE = styles['BodyText']
TABLE_STYLE =  TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ])
