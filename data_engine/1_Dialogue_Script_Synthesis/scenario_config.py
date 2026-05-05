"""
场景配置和维度定义模块
定义所有场景维度的枚举、分布权重
角色画像（名字、性格等）由 LLM 在第一阶段生成
"""

import os
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# ============================================================
# 情感标签映射
# ============================================================
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# ============================================================
# 场景维度定义 (选项, 权重)
# ============================================================

NUM_USERS_OPTIONS: List[Tuple[int, float]] = [
    (1, 0.02),
    (2, 0.80),
    (3, 0.08),
    (4, 0.05),
    (5, 0.01),
    (6, 0.01),
    (7, 0.01),
    (8, 0.01),
    (9, 0.01),
]

NUM_USERS_OPTIONS_ROLE: List[Tuple[int, float]] = [
    (1, 0.02),
    (2, 0.80),
    (3, 0.08),
    (4, 0.05),
    (5, 0.01),
    (6, 0.01),
    (7, 0.01),
    (8, 0.01),
    (9, 0.01),
]

# AI assistant 模式下，绝大多数为 1 人 vs AI 的对话；偶尔 2-3 人共享 AI 助手
NUM_USERS_OPTIONS_AI_ASSISTANT: List[Tuple[int, float]] = [
    (1, 0.85),
    (2, 0.10),
    (3, 0.05),
]


TURN_COUNTS: List[Tuple[Tuple[int, int], float]] = [
    ((50, 80), 0.1),    # 短对话
    ((80, 200), 0.8),   # 中等对话
    ((200, 300), 0.1),  # 长对话
]

# ============================================================
# 新增可变维度
# ============================================================

LANGUAGES: List[Tuple[str, float]] = [
    ("en", 0.5),
    ("zh", 0.5),
    ("mixed", 0.0),
]

LANGUAGE_INSTRUCTIONS = {
    "en": "All names, descriptions, and dialogue in English.",
    "zh": "All names, descriptions, and dialogue in Chinese (Mandarin). Use natural spoken Chinese, not translated English.",
    "mixed": (
        'Names can be Chinese or English. Descriptions in Chinese. '
        'Dialogue uses Chinese but naturally mixes in English words/phrases (code-switching), '
        'e.g. "这个project的deadline快到了", "我觉得这个idea还不错". '
        'The AI assistant responds in the same language the user uses. '
        'The mixing should feel natural, not forced.'
    ),
}

AI_INVOLVEMENT: List[Tuple[str, float]] = [
    ("high", 0.98),
    ("moderate", 0.02),
    # ("passive", 0.02),
]

AI_INVOLVEMENT_INSTRUCTIONS = {
    "high": (
        "The AI is an active participant — it joins the conversation naturally, "
        "shares opinions, asks questions, and responds to users at roughly the same "
        "frequency as the human users. It feels like a group chat member, not a service."
    ),
    "moderate": (
        "The AI chimes in every 2-3 user turns. It offers perspective when relevant, "
        "answers direct questions, and occasionally steers the conversation, but doesn't dominate."
    ),
    "passive": (
        "The AI mostly listens. It only speaks when directly addressed, asked a question, "
        "or when it has critical information to share. Most of the conversation is between "
        "the human users."
    ),
}

INDUSTRY_TOPICS = {

    # ── 科技与互联网 ──────────────────────────────────────────────
    "Early-Stage Tech Startup (Seed/Series A)": [
        "MVP Definition & Feature Scoping",
        "Fundraising Pitch Deck Preparation",
        "Co-founder Equity Split Dispute",
        "Hiring the First 10 Employees",
        "Product-Market Fit Discussion",
        "Runway & Burn Rate Review",
        "Pivot Decision — Changing Business Direction",
        "Tech Stack Selection Debate",
        "First Enterprise Client Onboarding Strategy",
        "Seed Round Term Sheet Negotiation Prep",
    ],

    "Growth-Stage SaaS Company (Series B/C)": [
        "Quarterly Business Review (QBR)",
        "Churn Rate Deep-Dive & Retention Strategy",
        "Sales & Marketing Alignment Meeting",
        "Pricing Model Overhaul (Freemium vs. Enterprise)",
        "Scaling Engineering Team — Hiring Plan",
        "Feature Prioritization: Product vs. Engineering Clash",
        "New Market Expansion Strategy",
        "Customer Success & Upsell Pipeline Review",
        "Series C Investor Update Preparation",
        "Annual Recurring Revenue (ARR) Forecast vs. Actuals",
    ],

    "Big Tech Corporation (FAANG-style)": [
        "Annual Performance Review Calibration",
        "Cross-Org Alignment on a Platform Initiative",
        "Layoff & Reorg Planning (Confidential)",
        "Internal AI Policy & Responsible Use Debate",
        "Antitrust & Regulatory Response Strategy",
        "Product Sunset Decision",
        "Promotion Committee Discussion",
        "Return-to-Office Policy Debate",
        "Data Privacy Compliance Review",
        "Acquisition Target Due Diligence",
    ],

    "AI / Machine Learning Research Lab": [
        "Research Roadmap Prioritization",
        "Model Evaluation & Benchmark Results Review",
        "Ethical AI & Bias Mitigation Discussion",
        "Paper Submission & Authorship Dispute",
        "Dataset Acquisition & Licensing Negotiation",
        "Commercialization vs. Open-Source Release Debate",
        "Safety Red-Teaming Results Debrief",
        "Compute Budget Allocation",
        "Talent Retention — Researcher Poaching by Industry",
        "Collaboration Agreement with External University",
    ],

    "Cybersecurity Firm": [
        "Active Data Breach Incident Response",
        "Vulnerability Disclosure Policy Debate",
        "Penetration Testing Report Client Debrief",
        "SOC 2 / ISO 27001 Certification Preparation",
        "Threat Intelligence Sharing Partnership",
        "Ransomware Attack Client Crisis Call",
        "Zero-Day Exploit Responsible Disclosure",
        "Annual Security Audit Review",
        "New Product Feature: Privacy vs. Functionality Tradeoff",
        "Government Contract Compliance Discussion",
    ],

    "Fintech Startup": [
        "Regulatory Licensing & Compliance Roadmap",
        "Banking-as-a-Service Partner Negotiation",
        "Fraud Detection Model Review",
        "Customer Onboarding KYC/AML Process Overhaul",
        "Product Launch: New Payment Feature",
        "Series A Pitch Deck Rehearsal",
        "Chargeback & Dispute Rate Crisis Meeting",
        "Cross-Border Expansion Regulatory Strategy",
        "App Redesign — UX vs. Compliance Clash",
        "Partnership with Traditional Bank Discussion",
    ],

    "Blockchain / Web3 Company": [
        "Token Economics (Tokenomics) Design Discussion",
        "Smart Contract Audit Results Review",
        "DAO Governance Structure Debate",
        "Exchange Listing Strategy Meeting",
        "Regulatory Uncertainty Response Planning",
        "NFT Project Roadmap & Community Strategy",
        "Treasury Management & Bear Market Survival",
        "Protocol Upgrade Proposal (Hard Fork Decision)",
        "Partnership with Traditional Finance Institution",
        "Community Crisis — Hack or Rug Pull Response",
    ],

    "Cloud Infrastructure Provider": [
        "Major Outage Post-Mortem",
        "New Data Center Region Expansion Decision",
        "Enterprise Customer SLA Negotiation",
        "Pricing Model Restructure",
        "Security Incident Response",
        "Competitor Pricing War Strategy",
        "Sustainability & Carbon Neutrality Commitment Planning",
        "Partner Ecosystem Strategy",
        "Service Deprecation Planning",
        "Annual Capacity Planning Review",
    ],

    "Hardware & IoT Company": [
        "Supply Chain Disruption Emergency Meeting",
        "Product Hardware Defect & Recall Decision",
        "Chip Shortage Impact on Production Timeline",
        "New Product Certification (FCC/CE) Planning",
        "Firmware Update Strategy & OTA Rollout",
        "Manufacturing Partner Quality Audit",
        "Bill of Materials (BOM) Cost Reduction",
        "Consumer Electronics Show (CES) Launch Prep",
        "IoT Security Vulnerability Disclosure",
        "Hardware vs. Software Roadmap Prioritization",
    ],

    "Software Outsourcing / IT Services Firm": [
        "Client Project Delivery Delay Crisis",
        "Resource Allocation Across Multiple Projects",
        "New Client Requirements Scoping",
        "Team Augmentation Proposal Negotiation",
        "Code Quality & Technical Debt Review",
        "Employee Attrition & Bench Management",
        "Transitioning from Fixed-Price to Time-and-Materials",
        "Offshore vs. Nearshore Team Restructuring",
        "Client Escalation & Relationship Recovery",
        "New Technology Practice Launch (e.g., AI/ML)",
    ],

    # ── 金融与投资 ────────────────────────────────────────────────
    "Venture Capital Firm": [
        "Investment Committee — New Deal Approval",
        "Portfolio Company Struggling — Follow-on or Write-off",
        "Fund II Fundraising Strategy",
        "LP Annual Update Preparation",
        "Co-investment Opportunity with Rival VC",
        "Term Sheet Negotiation Debrief",
        "Founder Reference Check Discussion",
        "Market Thesis Update — New Sector Focus",
        "Portfolio Company Exit Strategy (M&A or IPO)",
        "Junior Partner Promotion Decision",
    ],

    "Private Equity Firm": [
        "Leveraged Buyout (LBO) Target Evaluation",
        "100-Day Plan for Newly Acquired Company",
        "Portfolio Company Performance Review",
        "Exit Strategy: IPO vs. Strategic Sale",
        "Management Team Replacement Discussion",
        "Add-on Acquisition Strategy",
        "EBITDA Improvement Initiative Review",
        "Debt Covenant Breach Response",
        "Annual Fund Performance Presentation to LPs",
        "ESG Due Diligence Integration",
    ],

    "Hedge Fund": [
        "Morning Risk & P&L Review",
        "New Investment Thesis Debate",
        "Risk Limit Breach Discussion",
        "Investor Redemption Crisis Response",
        "Short Position Controversy (Activist Situation)",
        "Quant Model Drawdown Post-Mortem",
        "Prime Brokerage Relationship Review",
        "Macro Outlook & Portfolio Positioning",
        "Regulatory Investigation Response",
        "Analyst Hiring & Compensation Negotiation",
    ],

    "Investment Bank (M&A Division)": [
        "Pitch Book Review Before Client Meeting",
        "Deal Structuring — Merger Terms Debate",
        "Fairness Opinion Preparation",
        "Client Call Debrief — Hostile Takeover Defense",
        "Managing Director Deal Credit Dispute",
        "Cross-Border Deal Regulatory Hurdles",
        "Competing Bid Strategy",
        "IPO Roadshow Preparation",
        "Analyst Burnout & Staffing Crisis",
        "League Table Positioning Strategy",
    ],

    "Retail Bank / Commercial Bank": [
        "Non-Performing Loan (NPL) Review",
        "Regulatory Examination Preparation (FDIC/OCC)",
        "New Product Launch: Digital Savings Account",
        "Branch Closure Decision",
        "Anti-Money Laundering (AML) Compliance Review",
        "Credit Risk Policy Update",
        "Digital Transformation Strategy",
        "Customer Complaint Escalation",
        "Interest Rate Risk Management Discussion",
        "Fintech Partnership Evaluation",
    ],

    "Insurance Company": [
        "Catastrophic Claims Event Response (Natural Disaster)",
        "Actuarial Model Update & Premium Repricing",
        "New Product Design: Parametric Insurance",
        "Fraud Investigation Case Review",
        "Regulatory Solvency Compliance Review",
        "Digital Claims Processing Transformation",
        "Reinsurance Treaty Negotiation",
        "Customer Retention After Rate Increase",
        "Underwriting Policy Change Debate",
        "ESG Risk Integration into Portfolio",
    ],

    "Accounting & Auditing Firm": [
        "Audit Findings — Client Material Weakness Discussion",
        "Restatement of Financial Statements Crisis",
        "New Client Onboarding Risk Assessment",
        "Independence & Conflict of Interest Review",
        "Busy Season Staffing & Overtime Policy",
        "Partner Admission Decision",
        "Regulatory Inspection (PCAOB) Preparation",
        "New Revenue Recognition Standard Implementation",
        "Client Relationship Deterioration & Loss Risk",
        "Digital Audit Tool Adoption Debate",
    ],

    "Cryptocurrency Exchange": [
        "Regulatory Compliance Crisis (SEC Investigation)",
        "Major Security Breach & User Fund Protection",
        "New Token Listing Decision",
        "Liquidity Crisis & Market Maker Negotiation",
        "Customer Withdrawal Freeze Communication Strategy",
        "International Expansion Licensing Strategy",
        "Competitor Exchange Collapse Response",
        "Product Launch: Crypto Derivatives",
        "Bank Partnership Termination Crisis",
        "Community Trust Rebuilding After Controversy",
    ],

    "Wealth Management / Family Office": [
        "Client Portfolio Review — Major Market Downturn",
        "Ultra-High-Net-Worth Client Onboarding",
        "Estate Planning & Succession Strategy",
        "Alternative Investment Allocation Debate",
        "Client Complaint — Underperformance Discussion",
        "New Service Launch: Digital Wealth Platform",
        "Family Business Sale Advisory",
        "Tax Optimization Strategy Review",
        "Philanthropic Foundation Setup",
        "Senior Advisor Departure & Client Retention Plan",
    ],

    # ── 媒体、广告与创意 ──────────────────────────────────────────
    "Integrated Marketing Agency": [
        "Annual Campaign Strategy Presentation to Client",
        "Client Budget Cut Crisis Response",
        "Creative Brief Kickoff for New Account",
        "Pitch Competition Preparation",
        "Campaign Post-Mortem (Missed KPIs)",
        "New Business Development Strategy",
        "Talent Retention — Star Creative Leaving",
        "Rebranding Proposal for Long-Term Client",
        "Cross-Channel Attribution Debate",
        "Agency Merger / Acquisition Discussion",
    ],

    "Advertising Creative Studio": [
        "Creative Concept Presentation & Client Feedback",
        "Art Director vs. Copywriter Creative Clash",
        "Campaign Concept Rejection by Client",
        "Award Submission Strategy (Cannes / D&AD)",
        "Brand Voice & Identity Workshop",
        "Offensive Ad Controversy Crisis Response",
        "Freelance vs. In-House Team Debate",
        "New Client Onboarding Creative Kickoff",
        "Video Production Budget Overrun",
        "Creative Director Succession Planning",
    ],

    "Digital Content / Influencer Agency": [
        "Influencer Misconduct Crisis Management",
        "Content Calendar & Brand Partnership Negotiation",
        "Platform Algorithm Change Impact Review",
        "Creator Contract Dispute Resolution",
        "New Platform Strategy (e.g., Entering TikTok / Shorts)",
        "Brand Safety Incident Response",
        "Performance Metrics Review — Declining Engagement",
        "Exclusive Creator Signing Negotiation",
        "Campaign ROI Presentation to Client",
        "Creator vs. Agency Revenue Split Dispute",
    ],

    "Public Relations Firm": [
        "Crisis Communications — Client Scandal Eruption",
        "Media Training Session Preparation",
        "Press Release Review & Approval",
        "Journalist Relationship Strategy",
        "Client Reputation Rebuilding Plan",
        "New Account Pitch Strategy",
        "Social Media Monitoring Alert Response",
        "Executive Thought Leadership Campaign",
        "Product Launch Media Strategy",
        "Negative Article Mitigation Strategy",
    ],

    "News Media Outlet (Print / Online)": [
        "Editorial Meeting — Daily Story Lineup",
        "Breaking News Coverage Coordination",
        "Investigative Story Legal Review Before Publication",
        "Editorial Independence vs. Advertiser Pressure",
        "Layoff & Newsroom Restructuring Announcement",
        "Paywall & Subscription Strategy Review",
        "Source Protection & Ethics Discussion",
        "Viral Misinformation Response",
        "New Digital Platform Strategy",
        "Reporter's Story Pulled — Internal Dispute",
    ],

    "Broadcast Television Network": [
        "Primetime Ratings Crisis Strategy",
        "New Season Commissioning Decisions",
        "On-Air Talent Contract Renegotiation",
        "Live Broadcast Error Crisis Response",
        "Streaming vs. Linear TV Strategy Debate",
        "Diversity & Representation in Programming",
        "Advertiser Boycott Response",
        "Reality Show Controversy Management",
        "News Division Editorial Integrity Debate",
        "International Format Licensing Deal",
    ],

    "Podcast Production Company": [
        "Host Departure Crisis & Show Continuity",
        "Sponsorship Deal Negotiation",
        "New Show Greenlight Decision",
        "Listener Growth Strategy Review",
        "Controversial Episode — Publish or Pull Decision",
        "Exclusive Distribution Deal (Spotify / Apple) Debate",
        "Cross-Promotion Partnership Planning",
        "Monetization Strategy Overhaul",
        "Production Quality vs. Budget Tradeoff",
        "Brand Extension: Video Podcast Launch",
    ],

    "Film Production Company": [
        "Greenlight Decision for New Film Project",
        "Budget Overrun Crisis on Active Production",
        "Director vs. Studio Creative Control Clash",
        "Casting Controversy & Public Backlash",
        "Distribution Deal Negotiation (Theatrical vs. Streaming)",
        "Film Festival Strategy (Cannes / Sundance)",
        "Re-shoot Decision After Poor Test Screening",
        "Talent Misconduct Response & Production Impact",
        "International Co-production Agreement",
        "Marketing & Awards Campaign Strategy",
    ],

    "TV Drama / Streaming Series Production": [
        "Writers Room — Season Story Arc Debate",
        "Showrunner & Network Creative Clash",
        "Production Shutdown Crisis (Strike / Accident)",
        "Cast Member Contract Dispute",
        "Season Renewal or Cancellation Discussion",
        "Post-Production Schedule & VFX Budget Overrun",
        "Streaming Platform Note Review (Creative vs. Commercial)",
        "Diversity in Writers Room Initiative",
        "International Adaptation Rights",
        "Episode Release Strategy: Binge vs. Weekly",
    ],

    "Animation Studio": [
        "Animation Style & Art Direction Sign-off",
        "Pipeline & Software Migration Decision",
        "Feature Film vs. Series Resource Allocation",
        "Voice Casting Decision",
        "Co-production Deal with International Partner",
        "Studio IP Licensing Strategy",
        "Animation Quality vs. Deadline Debate",
        "New Technology Adoption: AI Animation Tools",
        "Franchise Extension Planning",
        "Award Season Campaign (Oscar / Annie Awards)",
    ],

    "Video Game Development Studio (AAA)": [
        "Game Crunch Policy & Developer Wellbeing",
        "Live Service Monetization Strategy Debate",
        "Game Delay Decision — Quality vs. Release Date",
        "Loot Box & Microtransaction Controversy Response",
        "Post-Launch Patch & Bug Fix Priority Meeting",
        "Next-Gen Console Port Planning",
        "Community Backlash Response Strategy",
        "DLC & Season Pass Planning",
        "Studio Culture & Harassment Policy Review",
        "IP Licensing & Franchise Expansion",
    ],

    "Indie Game Studio": [
        "Kickstarter Campaign Strategy",
        "Publisher Deal vs. Self-Publishing Debate",
        "Scope Creep & Feature Cut Decision",
        "Early Access Launch Strategy",
        "Game Jam Prototype Review",
        "Team Burnout & Sustainability Discussion",
        "Art Style Pivot Decision",
        "Port to New Platform (Mobile / Console)",
        "Community Feedback Integration",
        "Post-Launch Revenue vs. Expectations Debrief",
    ],

    # ── 医疗与生命科学 ────────────────────────────────────────────
    "Public Hospital (Clinical Department)": [
        "ICU Capacity Crisis Management",
        "Medical Error & Patient Safety Incident Review",
        "Department Budget Cuts Impact Discussion",
        "New Clinical Protocol Adoption",
        "Staffing Shortage & Nurse Retention Crisis",
        "Patient Complaint Escalation",
        "Multi-Disciplinary Team (MDT) Case Conference",
        "Elective Surgery Backlog Reduction Strategy",
        "Infection Control Outbreak Response",
        "Junior Doctor Working Hours Compliance",
    ],

    "Private Healthcare Clinic": [
        "Patient Satisfaction Score Review",
        "New Service Launch (e.g., Aesthetic Medicine)",
        "Insurance Reimbursement Dispute",
        "Doctor Partnership Buy-In Negotiation",
        "Staff Misconduct Investigation",
        "Digital Health Records System Migration",
        "Marketing & Patient Acquisition Strategy",
        "Clinic Expansion Planning",
        "Medical Equipment Procurement Review",
        "Pricing Strategy for Out-of-Pocket Services",
    ],

    "Pharmaceutical R&D Company": [
        "Clinical Trial Phase III Results Review",
        "FDA / EMA Submission Strategy",
        "Drug Safety Signal Investigation",
        "Pipeline Prioritization — Which Programs to Cut",
        "Patent Expiry & Generic Competition Strategy",
        "Research Partnership with Academic Institution",
        "Manufacturing Scale-Up Decision",
        "Pricing & Market Access Strategy for New Drug",
        "Post-Market Surveillance Data Review",
        "Competitor Drug Approval Impact Assessment",
    ],

    "Biotech Startup": [
        "Series B Fundraising Pitch Preparation",
        "Preclinical Data Review Before IND Filing",
        "Platform Technology Licensing Strategy",
        "Scientific Advisory Board Input Session",
        "Partnership with Big Pharma — Term Negotiation",
        "Regulatory Strategy (Breakthrough Therapy Designation)",
        "IP & Patent Filing Strategy",
        "Key Scientist Departure & Knowledge Retention",
        "Animal Study Results Debrief",
        "Go / No-Go Decision for Lead Compound",
    ],

    "Medical Device Manufacturer": [
        "Product Recall Crisis & Regulatory Notification",
        "FDA 510(k) Clearance Strategy",
        "Post-Market Clinical Study Design",
        "Hospital Key Opinion Leader (KOL) Engagement",
        "Manufacturing Quality Complaint Investigation",
        "Competitive Device Launch Response",
        "Sales Force Training on New Device",
        "International CE Marking Strategy",
        "Hospital Procurement & Tender Strategy",
        "Software as a Medical Device (SaMD) Compliance",
    ],

    "Health Insurance Provider": [
        "Claims Denial Rate Investigation",
        "New Plan Design for Employer Group",
        "Provider Network Renegotiation",
        "Fraud, Waste & Abuse Case Review",
        "Regulatory Compliance (ACA Requirements)",
        "Member Retention After Premium Increase",
        "Prior Authorization Policy Debate",
        "Digital Health App Partnership Evaluation",
        "Medical Loss Ratio (MLR) Review",
        "Mental Health Parity Compliance",
    ],

    "Mental Health Services Organization": [
        "Crisis Intervention Protocol Review",
        "Therapist Burnout & Caseload Management",
        "Insurance Reimbursement & Funding Shortfall",
        "New Digital Therapy Platform Adoption",
        "Patient Suicide Risk Assessment Policy Update",
        "Community Outreach Program Planning",
        "Staff Clinical Supervision Meeting",
        "Diversity-Informed Care Initiative",
        "Waitlist Crisis — Demand vs. Capacity",
        "Grant Application Strategy",
    ],

    "Clinical Research Organization (CRO)": [
        "Site Initiation Visit (SIV) Readiness Review",
        "Protocol Deviation Investigation",
        "Sponsor Audit Preparation",
        "Patient Recruitment Crisis — Enrollment Delay",
        "Data Integrity & GCP Compliance Discussion",
        "Budget vs. Scope Change Negotiation with Sponsor",
        "Adverse Event Reporting Escalation",
        "Trial Master File (TMF) Inspection Readiness",
        "Multi-Site Coordination Challenges",
        "Transition to Risk-Based Monitoring",
    ],

    # ── 消费品与零售 ──────────────────────────────────────────────
    "E-commerce Platform (Marketplace)": [
        "Seller Policy Violation & Suspension Decision",
        "Peak Season (11.11 / Black Friday) Readiness",
        "Logistics Partner Performance Review",
        "Fake Review & Fraud Detection Strategy",
        "New Category Expansion Planning",
        "Seller Fee Structure Overhaul",
        "Cross-Border E-commerce Regulation Compliance",
        "Customer Return Rate Crisis",
        "Live Commerce Strategy Launch",
        "Algorithm Change Impact on Seller GMV",
    ],

    "D2C Fashion Brand": [
        "New Collection Design Review",
        "Influencer Campaign Performance Debrief",
        "Sustainability & Ethical Sourcing Strategy",
        "Inventory Overstock Crisis",
        "Pop-Up Store Event Planning",
        "Customer Acquisition Cost (CAC) vs. LTV Review",
        "Collaboration with External Designer / Brand",
        "Social Media Controversy Response",
        "Factory Quality Issue & Shipment Delay",
        "Wholesale vs. DTC Channel Strategy Debate",
    ],

    "Luxury Goods Company": [
        "Brand Exclusivity vs. Revenue Growth Tension",
        "Counterfeiting & Brand Protection Strategy",
        "New Market Entry: China / Middle East",
        "Celebrity Ambassador Contract Review",
        "Digital Strategy While Preserving Luxury Image",
        "Craftsmanship & Supplier Relationship Review",
        "Store Design & Retail Experience Overhaul",
        "Vintage & Resale Market Response",
        "Heritage Collection Relaunch",
        "ESG & Sustainability Reporting",
    ],

    "Fast Fashion Retailer": [
        "Sustainability Greenwashing Controversy Response",
        "Supply Chain Labor Conditions Investigation",
        "Trend Forecasting & Buying Strategy",
        "Unsold Inventory Write-Down Decision",
        "E-commerce vs. Physical Store Strategy",
        "New Market Entry Strategy",
        "Influencer & Social Media Campaign Planning",
        "Cost Reduction: Factory Relocation Decision",
        "Customer Return Policy Overhaul",
        "Circular Fashion Initiative Launch",
    ],

    "Cosmetics & Beauty Brand": [
        "New Product Development & Ingredient Safety Review",
        "Shade Range Inclusivity Controversy Response",
        "Influencer Partnership Strategy",
        "Clean Beauty Reformulation Initiative",
        "Retailer Shelf Space Negotiation (Sephora / Ulta)",
        "Brand Ambassador Misconduct Crisis",
        "Animal Testing Policy Update",
        "International Regulatory Compliance (EU Cosmetics)",
        "Limited Edition Launch Planning",
        "Competitor Brand Acquisition Discussion",
    ],

    "Food & Beverage Company": [
        "Product Contamination & Recall Crisis",
        "New Product Development Taste Test Review",
        "Retail Shelf Placement Negotiation",
        "Supply Chain Cost Inflation Response",
        "Health Claim Regulatory Compliance",
        "Brand Refresh & Packaging Redesign",
        "International Market Entry Strategy",
        "ESG: Reducing Sugar / Plastic Packaging Initiative",
        "Franchise System Performance Review",
        "Social Media Viral Moment Response",
    ],

    "Supermarket / Grocery Chain": [
        "Supplier Contract Renegotiation",
        "Private Label vs. National Brand Strategy",
        "Store Layout & Planogram Overhaul",
        "Food Waste Reduction Initiative",
        "Online Grocery & Delivery Strategy",
        "Price War Response to Competitor",
        "New Store Opening Decision",
        "Product Recall Notification Protocol",
        "Staff Scheduling & Labor Cost Review",
        "Loyalty Program Redesign",
    ],

    "Consumer Electronics Brand": [
        "New Product Hardware Design Sign-off",
        "Supply Chain Disruption — Component Shortage",
        "Product Launch Event Planning",
        "Software Update Causing Device Issues Crisis",
        "Pricing Strategy vs. Competitor",
        "Retail Partnership vs. Direct Sales Debate",
        "Customer Repair & Right-to-Repair Policy",
        "Battery Safety Concern Investigation",
        "Sustainability: E-waste & Recycling Program",
        "Next-Generation Product Roadmap Review",
    ],

    # ── 专业服务 ──────────────────────────────────────────────────
    "Management Consulting Firm (Big 4-style)": [
        "Client Engagement Kickoff — Setting Scope & Expectations",
        "Partner Review of Associate / Analyst Work",
        "Client Relationship at Risk — Escalation Meeting",
        "Proposal Review Before Submission",
        "Practice Area Strategy & Market Positioning",
        "Diversity & Inclusion Initiative Discussion",
        "Up-or-Out Promotion Decision",
        "Conflict of Interest Check on New Client",
        "Knowledge Management & IP Sharing",
        "Cross-Selling Services to Existing Client",
    ],

    "Strategy Consulting Boutique": [
        "New Client Pitch Preparation",
        "Engagement Findings Presentation Rehearsal",
        "Scope Creep & Fee Negotiation with Client",
        "Hiring Strategy for Specialized Expertise",
        "Partner Compensation & Profit-Sharing Debate",
        "Niche Market Positioning Strategy",
        "Retaining vs. Replacing a Difficult Client",
        "Building a Proprietary Methodology",
        "Associate Performance & Mentorship Discussion",
        "Boutique vs. Big 4 Competitive Positioning",
    ],

    "Law Firm (Corporate / Litigation)": [
        "Case Strategy Meeting — High-Stakes Litigation",
        "Partner Compensation Review",
        "Client Conflict of Interest Check",
        "Billing Write-Down Negotiation with Client",
        "Associate Performance & Partnership Track Discussion",
        "New Client Pitch for Major Corporation",
        "Pro Bono Commitment & Firm Policy",
        "Merger with Another Firm — Due Diligence",
        "Lateral Partner Hire Decision",
        "Practice Group Expansion Strategy",
    ],

    "Headhunting / Executive Recruitment Firm": [
        "Senior Executive Search Candidate Shortlist Review",
        "Client Briefing on CHRO Search Requirements",
        "Candidate Counter-Offer Crisis Management",
        "Fee Dispute with Client",
        "Exclusive vs. Retained Search Model Debate",
        "Candidate Reference Check Discussion",
        "Diversity Placement Target Review",
        "New Industry Vertical Launch Strategy",
        "Consultant Commission & Bonus Dispute",
        "Digital Platform vs. Traditional Headhunting Debate",
    ],

    "Architecture & Urban Design Studio": [
        "Design Concept Presentation to Client",
        "Planning Permission Rejection Response",
        "Project Budget Overrun Discussion",
        "Sustainability & Green Building Certification",
        "Award Submission Strategy (Pritzker / RIBA)",
        "Client Vision vs. Structural Feasibility Clash",
        "New Landmark Project Pitch",
        "Studio Hiring & Portfolio Diversity",
        "Construction Partner Collaboration Issues",
        "Historic Building Renovation Controversy",
    ],

    "Interior Design Agency": [
        "Client Design Brief Review & Mood Board Approval",
        "Supplier & Furniture Procurement Delay",
        "Client Change Request & Budget Impact",
        "Residential vs. Commercial Project Prioritization",
        "Design Concept Rejection & Pivot",
        "Photography & Portfolio Shoot Planning",
        "New Client Onboarding & Expectation Setting",
        "Sustainability Material Selection Discussion",
        "Project Handover & Punch List Review",
        "Award Entry & PR Strategy",
    ],

    # ── 工业、制造与物流 ──────────────────────────────────────────
    "Automobile Manufacturer": [
        "EV Transition Strategy & ICE Phase-Out Timeline",
        "Factory Safety Incident Investigation",
        "Supply Chain Shortage (Semiconductor Chips)",
        "Recall Decision — Safety vs. Cost",
        "New Model Launch Planning",
        "Union Labor Negotiation Preparation",
        "Dealership Network Performance Review",
        "Autonomous Driving Feature Regulation Compliance",
        "Joint Venture with Foreign Manufacturer",
        "Carbon Emission Reduction Target Review",
    ],

    "Semiconductor Fabrication Plant": [
        "Yield Rate Drop Root Cause Analysis",
        "New Node Process Development Decision",
        "Equipment Procurement & Capex Approval",
        "Geopolitical Risk & Supply Chain Diversification",
        "Customer Allocation During Chip Shortage",
        "Cleanroom Contamination Incident Response",
        "IP Theft Allegation Investigation",
        "Government Subsidy Application Strategy",
        "Workforce Safety & Chemical Hazard Review",
        "Next-Generation Chip Architecture Planning",
    ],

    "Aerospace & Defense Contractor": [
        "Government Contract Bid Strategy",
        "Program Cost Overrun & Schedule Delay",
        "Export Control & ITAR Compliance Review",
        "Prototype Test Failure Debrief",
        "Classified Program Access Control Meeting",
        "Subcontractor Performance Issue",
        "New Defense Technology R&D Proposal",
        "Safety Review Board — Flight Test Incident",
        "International Partnership Agreement",
        "Transition from Legacy to Next-Gen Platform",
    ],

    "Chemical Manufacturing Company": [
        "Plant Safety Incident Root Cause Analysis",
        "Environmental Regulation Compliance Review",
        "New Chemical Product Development",
        "Supply Chain Raw Material Shortage",
        "Customer Specification Change Impact",
        "Hazardous Waste Disposal Controversy",
        "ISO 14001 Audit Preparation",
        "Worker Health & Exposure Limit Review",
        "R&D Investment Prioritization",
        "Export Restriction & Trade Compliance",
    ],

    "Logistics & Supply Chain Corporation": [
        "Peak Season Capacity Planning",
        "Last-Mile Delivery Performance Review",
        "New Warehouse Automation Investment Decision",
        "Driver Shortage & Labor Cost Crisis",
        "Customer SLA Breach Escalation",
        "Cross-Border Customs Compliance Issue",
        "Carbon Emission & Green Logistics Initiative",
        "Technology Partner Selection (WMS / TMS)",
        "E-commerce Client Onboarding",
        "Route Optimization Strategy",
    ],

    "Shipping & Freight Company": [
        "Port Congestion Crisis Response",
        "Freight Rate Negotiation with Key Client",
        "Vessel Maintenance & Drydock Scheduling",
        "Crew Welfare & Maritime Regulation Compliance",
        "New Trade Route Feasibility Study",
        "Cargo Loss Investigation",
        "Fuel Efficiency & Decarbonization Strategy",
        "Piracy Risk Assessment for New Routes",
        "Digital Freight Platform Adoption",
        "Customer Cargo Damage Claim Resolution",
    ],

    "Construction & Real Estate Developer": [
        "Site Safety Incident Review",
        "Project Delay & Contractor Dispute",
        "Planning Permission Strategy for New Development",
        "Cost Overrun & Value Engineering Decision",
        "Investor Update on Development Portfolio",
        "Green Building & LEED Certification Planning",
        "Tenant Mix Strategy for Mixed-Use Development",
        "Interest Rate Impact on Development Financing",
        "Subcontractor Insolvency Crisis",
        "Community Consultation & Opposition Response",
    ],

    "Real Estate Agency (Residential / Commercial)": [
        "Market Slowdown & Team Motivation Strategy",
        "High-Value Property Listing Pitch",
        "Agent Commission Dispute",
        "New Office Opening Strategy",
        "Digital Marketing & Lead Generation Review",
        "Client Complaint & Deal Gone Wrong",
        "Luxury Property Launch Event Planning",
        "Commercial Lease Negotiation Strategy",
        "Agent Performance Review & Target Setting",
        "International Buyer Strategy",
    ],

    "Energy Company (Oil, Gas, or Renewables)": [
        "Renewable Energy Transition Roadmap",
        "Offshore Rig Safety Incident Response",
        "Oil Price Shock & Hedging Strategy",
        "New Wind / Solar Farm Project Approval",
        "Environmental Impact Assessment Review",
        "Government Regulation & Carbon Tax Impact",
        "Joint Venture Partner Negotiation",
        "Energy Grid Stability & Storage Solution",
        "Community & Indigenous Land Rights Consultation",
        "Stranded Asset Write-Down Decision",
    ],

    "Utility & Power Grid Company": [
        "Grid Outage Post-Mortem & Resilience Review",
        "Smart Grid Technology Investment Decision",
        "Regulatory Price Review Preparation",
        "EV Charging Infrastructure Expansion",
        "Wildfire Risk Mitigation Strategy",
        "Nuclear Plant Decommissioning Planning",
        "Renewable Integration & Intermittency Challenge",
        "Customer Billing System Overhaul",
        "Cybersecurity Threat to Critical Infrastructure",
        "Carbon Net-Zero Commitment Strategy",
    ],

    # ── 教育 ──────────────────────────────────────────────────────
    "K-12 Private School (Administrative Team)": [
        "Student Disciplinary Case Review",
        "Faculty Performance & Tenure Decision",
        "Curriculum Redesign Initiative",
        "Tuition Fee Increase Communication Strategy",
        "Parent Complaint Escalation",
        "Diversity & Inclusion Policy Update",
        "Admissions Strategy & Enrollment Targets",
        "Budget Allocation for New Facilities",
        "Student Mental Health Support Resources",
        "Accreditation Review Preparation",
    ],

    "University / Higher Education Institution": [
        "Research Grant Application Strategy",
        "Faculty Misconduct Investigation",
        "Student Mental Health Crisis Response",
        "Curriculum Modernization Debate",
        "International Student Enrollment Strategy",
        "Campus Free Speech & Academic Freedom Debate",
        "Endowment Investment Policy Review",
        "Diversity in Faculty Hiring",
        "Online Learning vs. In-Person Strategy",
        "Partnership with Industry for Research Funding",
    ],

    "EdTech Startup": [
        "Product Roadmap: B2C vs. B2B Pivot Decision",
        "Learning Outcome Data Review",
        "School District Partnership Negotiation",
        "Teacher / Student Onboarding Experience Overhaul",
        "Content Curriculum Quality Assurance",
        "Monetization Strategy: Subscription vs. Freemium",
        "AI Tutor Feature Launch Planning",
        "Data Privacy & COPPA Compliance (Children's Data)",
        "International Market Expansion",
        "Series A Fundraising Pitch Rehearsal",
    ],

    "Corporate Training & L&D Department": [
        "Annual Learning Strategy & Budget Planning",
        "Leadership Development Program Design",
        "Mandatory Compliance Training Rollout",
        "LMS Platform Migration Decision",
        "Measuring Training ROI & Effectiveness",
        "External Training Vendor Selection",
        "Upskilling for AI / Digital Transformation",
        "New Employee Onboarding Program Redesign",
        "Manager Capability Assessment Results",
        "DEI Training Program Evaluation",
    ],

    "Test Prep & Tutoring Company": [
        "Curriculum Update for New Exam Format",
        "Instructor Performance & Quality Review",
        "Student Outcome & Pass Rate Analysis",
        "Online vs. In-Person Class Strategy",
        "New Market Entry (New City / Country)",
        "Partnership with School for Referrals",
        "Pricing & Scholarship Strategy",
        "AI-Powered Practice Test Feature Launch",
        "Brand Awareness & Marketing Campaign",
        "Franchise Model Expansion Decision",
    ],

    # ── 公共部门与非营利 ──────────────────────────────────────────
    "Government Agency (Policy / Regulatory Body)": [
        "New Regulation Drafting & Public Consultation",
        "Enforcement Action Against Major Corporation",
        "Inter-Agency Coordination on Policy Issue",
        "Budget Allocation for Next Fiscal Year",
        "Crisis Response: Natural Disaster / Public Health",
        "Whistleblower Complaint Investigation",
        "Digital Government Services Transformation",
        "International Treaty Compliance Review",
        "Parliamentary / Congressional Testimony Preparation",
        "Policy Impact Assessment Review",
    ],

    "City / Municipal Government Office": [
        "Infrastructure Project Approval & Funding",
        "Community Development Complaint Response",
        "City Budget Deficit Response Plan",
        "Public Transport Expansion Strategy",
        "Homelessness Crisis Response Planning",
        "Climate Action Plan Implementation",
        "Local Business Support Initiative",
        "Zoning Law Change Public Consultation",
        "Emergency Service Response Review",
        "New Mayor / Leadership Transition Planning",
    ],

    "International NGO": [
        "Field Program Performance Review",
        "Donor Relations & Funding Renewal Strategy",
        "Humanitarian Crisis Emergency Response",
        "Staff Safety & Security Protocol Update",
        "Program Impact Assessment",
        "Headquarters vs. Field Office Tension",
        "New Country Program Launch",
        "Grant Compliance & Reporting Crisis",
        "Partnership with UN Agency or Government",
        "Internal Fraud Investigation",
    ],

    "Domestic Nonprofit Organization": [
        "Annual Fundraising Campaign Strategy",
        "Board Governance & Leadership Transition",
        "Program Cuts Due to Funding Shortfall",
        "Volunteer Management & Retention",
        "Community Needs Assessment Findings",
        "Grant Application & Reporting",
        "Beneficiary Complaint & Safeguarding Review",
        "Partnership with Corporate Sponsor",
        "Social Media & Awareness Campaign Planning",
        "Strategic Plan Review & New Mission Alignment",
    ],

    "Think Tank / Policy Research Institute": [
        "Research Report Launch & Media Strategy",
        "Funding Source Independence & Bias Concerns",
        "Policy Brief Review Before Government Submission",
        "Expert Advisory Panel Coordination",
        "Annual Research Agenda Planning",
        "Public Event & Conference Organization",
        "Researcher Recruitment & Retention",
        "Cross-Institute Collaboration Agreement",
        "Media Controversy Over Published Findings",
        "New Donor Engagement Strategy",
    ],

    "United Nations or Intergovernmental Body": [
        "Security Council Resolution Drafting",
        "Humanitarian Mission Mandate Renewal",
        "Member State Non-Compliance Discussion",
        "Budget & Staff Reduction Crisis",
        "Peacekeeping Operation Review",
        "Climate Agreement Implementation Progress",
        "Sexual Misconduct Allegation Response",
        "New Program Launch in Fragile State",
        "Media & Public Communication Strategy",
        "Reform Proposal — Institutional Restructuring",
    ],

    # ── 餐饮、酒店与旅游 ──────────────────────────────────────────
    "Michelin-Starred Restaurant Group": [
        "New Menu Concept Tasting & Chef Debate",
        "Michelin Star Retention Strategy",
        "Front-of-House & Back-of-House Tension",
        "Staff Turnover & Hiring in Competitive Market",
        "New Restaurant Opening Location Decision",
        "Food Cost & Margin Review",
        "Celebrity Chef Partnership Negotiation",
        "Food Safety Incident Response",
        "Reservations System Overhaul",
        "Restaurant Week & PR Event Planning",
    ],

    "Hotel Chain (Corporate HQ)": [
        "Guest Satisfaction Score Review",
        "Property Renovation Investment Decision",
        "Revenue Management & Dynamic Pricing Strategy",
        "Franchise Agreement Dispute with Operator",
        "New Hotel Brand Segment Launch",
        "Loyalty Program Overhaul",
        "OTA Commission Negotiation (Booking.com / Expedia)",
        "Staff Retention in Competitive Labor Market",
        "Sustainability Certification Initiative",
        "Crisis Response: Viral Guest Complaint",
    ],

    "Airline Company": [
        "Flight Cancellation Crisis Communication",
        "Pilot Shortage & Scheduling Crisis",
        "New Route Launch Decision",
        "Fuel Hedging Strategy Review",
        "Passenger Overbooking Incident Response",
        "Fleet Modernization & Aircraft Procurement",
        "Frequent Flyer Program Devaluation Communication",
        "Union Negotiation & Strike Threat Response",
        "Safety Audit & Regulatory Compliance",
        "Alliance Partnership Renegotiation",
    ],

    "Travel & Tourism Platform": [
        "Peak Season Platform Stability Review",
        "Supplier Commission Dispute (Hotel / Airline)",
        "New Product Launch: Experiences Category",
        "Negative Review & Trust and Safety Policy",
        "Post-Crisis Recovery Strategy",
        "Sustainable Travel Initiative Launch",
        "International Expansion Regulatory Issues",
        "Fraud & Fake Listing Crisis",
        "Influencer Travel Campaign Planning",
        "Competitive Response to New Market Entrant",
    ],

    "Event Planning & MICE Company": [
        "Major Event Crisis — Venue Cancellation",
        "Client Budget Cut & Scope Reduction",
        "Event Day Logistics & Vendor Coordination",
        "Post-Event Client Debrief",
        "New Flagship Event Concept Pitch",
        "Hybrid (Virtual + In-Person) Event Strategy",
        "Sponsorship Acquisition Strategy",
        "Supplier Contract Negotiation",
        "Team Capacity Planning for Peak Season",
        "International Event Expansion Planning",
    ],

    # ── 体育与娱乐 ────────────────────────────────────────────────
    "Professional Sports Club (Management)": [
        "Transfer Window Strategy & Player Acquisition",
        "Head Coach Performance & Contract Renewal",
        "Stadium Naming Rights Sponsorship Deal",
        "Player Misconduct & Disciplinary Action",
        "Financial Fair Play Compliance Review",
        "Fan Engagement & Season Ticket Sales Strategy",
        "Youth Academy Development Investment",
        "Media Rights Negotiation",
        "Injury Crisis & Squad Depth Assessment",
        "Club Rebranding or Ownership Change Communication",
    ],

    "Sports Agency": [
        "Athlete Contract Negotiation Strategy",
        "Endorsement Deal Evaluation & Brand Fit",
        "Athlete Misconduct Crisis Management",
        "Post-Retirement Career Planning for Athlete",
        "New Athlete Signing — Pitch to Athlete",
        "Salary Arbitration Preparation",
        "Social Media Strategy for Athlete Brand",
        "Injury Impact on Endorsement Obligations",
        "Conflict Between Athlete and Team",
        "Agency Expansion into New Sport",
    ],

    "Music Label": [
        "Artist Album Release Strategy",
        "Streaming vs. Physical Revenue Discussion",
        "New Artist Signing Decision",
        "Artist Royalty Dispute",
        "Music Video Budget Approval",
        "Tour & Live Event Planning",
        "Artist & Label Creative Direction Clash",
        "Sync Licensing Opportunity Evaluation",
        "Social Media Viral Moment Monetization",
        "Label Merger or Acquisition Discussion",
    ],

    "Talent & Artist Management Agency": [
        "Client Brand Deal Negotiation",
        "Managing a Client Public Relations Crisis",
        "Contract Renewal vs. Competitor Poaching",
        "New Talent Discovery & Onboarding",
        "Cross-Media Expansion for Client (TV, Film, Brand)",
        "Client Burnout & Mental Health Discussion",
        "Revenue Commission Dispute",
        "Social Media Strategy for Rising Client",
        "Event Appearance Fee Negotiation",
        "Agency Rebranding & Market Positioning",
    ],

    "Theme Park / Entertainment Resort": [
        "New Attraction Concept Approval",
        "Guest Safety Incident Response",
        "Peak Season Crowd Management Strategy",
        "Annual Pass Pricing Strategy",
        "IP Licensing for New Themed Area",
        "Staff Hiring & Seasonal Workforce Planning",
        "Guest Satisfaction & Net Promoter Score Review",
        "International Resort Expansion Feasibility",
        "Environmental Sustainability Initiative",
        "Digital & App Experience Enhancement",
    ],
}

INDUSTRY_TOPICS_LIST = [(industry, topic) for industry, topics in INDUSTRY_TOPICS.items() for topic in topics]

# ============================================================
# 角色数据加载
# ============================================================

def load_roles_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    从 JSONL 文件加载角色数据

    每行格式: {"id": "...", "role": {...}} 或 {"id": "...", "role": [{...}]}
    返回角色对象列表
    """
    roles = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading roles from JSONL"):
            line = line.strip()
            if line:
                data = json.loads(line)
                role = data.get('role')
                if role:
                    # 处理 role 是列表的情况（取第一个）
                    if isinstance(role, list):
                        if len(role) > 0:
                            role = role[0]
                    role["id"] = data["id"]
                    roles.append(role)
    return roles


def load_character_data(jsonl_path: str) -> List[Dict]:
    """从 JSONL 文件加载角色数据（兼容旧接口）"""
    characters = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading character data from JSONL"):
            line = line.strip()
            if line:
                data = json.loads(line)
                # 提取 processed_role 字段
                if 'processed_role' in data:
                    characters.append(data['processed_role'])
    return characters


def sample_character(characters: List[Dict]) -> Dict:
    """从角色列表中随机选择一个角色"""
    return random.choice(characters)


def _weighted_choice(options: List[Tuple]) -> any:
    """从 (value, weight) 列表中按权重随机选择"""
    values, weights = zip(*options)
    return random.choices(values, weights=weights, k=1)[0]


def _weighted_choice_from_dict(weight_dict: Dict[str, float]) -> str:
    """从 {key: weight} 字典中按权重随机选择"""
    keys = list(weight_dict.keys())
    weights = [weight_dict[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


@dataclass
class ScenarioConfig:
    """采样得到的维度参数（不含角色画像，角色由 LLM 生成）"""
    seed_sentence: str
    seed_emotion: str
    num_users: int
    turn_count: int
    language: str
    ai_involvement: str
    predefined_assistant_role: Optional[Dict] = None  # 预定义的助手角色（可选）
    is_work_meeting: bool = False  # 是否为工作会议场景
    is_ai_assistant: bool = False  # 是否为 AI assistant 模式（assistant 是真实 AI 而非伪装人类）
    industry: Optional[str] = None  # 行业（可选）
    topic: Optional[str] = None  # 话题（可选）
    # --seed-qa：从外部 Q&A 数据集（默认 extracted_data.jsonl）抽一条作为
    # 对话话题"种子"。LLM 不必逐字复述，只把它当作灵感。为 None 时不注入。
    qa_question: Optional[str] = None
    qa_answer: Optional[str] = None


def random_qa_sample(jsonl_path: str, n: int, rng: random.Random) -> List[Dict]:
    """从大 Q&A JSONL 里随机 seek 抽 n 条记录（按 idx 去重）。

    与 super_assistant/sampler.py:_random_qa_sample 行为一致。
    """
    file_size = os.path.getsize(jsonl_path)
    if file_size == 0:
        return []
    out: List[Dict] = []
    seen: set = set()
    max_attempts = max(n * 5, 50)
    attempts = 0
    with open(jsonl_path, "rb") as f:
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            offset = rng.randint(0, max(0, file_size - 1))
            f.seek(offset)
            f.readline()  # 丢掉随机点切到的 partial 行
            line = f.readline()
            if not line:
                f.seek(0)
                line = f.readline()
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            key = rec.get("idx")
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append(rec)
    return out


def sample_scenario(seed_sentence: str, seed_emotion_label: int, assigned_role: bool = False, is_work_meeting: bool = False, is_ai_assistant: bool = False, industry: Optional[str] = None, topic: Optional[str] = None) -> ScenarioConfig:
    """从所有维度中采样一个完整的场景配置"""
    emotion = EMOTION_LABELS[seed_emotion_label]

    if is_ai_assistant:
        num_users = _weighted_choice(NUM_USERS_OPTIONS_AI_ASSISTANT)
    elif assigned_role:
        num_users = _weighted_choice(NUM_USERS_OPTIONS_ROLE)
    else:
        num_users = _weighted_choice(NUM_USERS_OPTIONS)

    turn_range = _weighted_choice(TURN_COUNTS)
    turn_count = random.randint(turn_range[0], turn_range[1])

    language = _weighted_choice(LANGUAGES)
    ai_inv = _weighted_choice(AI_INVOLVEMENT)

    return ScenarioConfig(
        seed_sentence=seed_sentence,
        seed_emotion=emotion,
        num_users=num_users,
        turn_count=turn_count,
        language=language,
        ai_involvement=ai_inv,
        is_work_meeting=is_work_meeting,
        is_ai_assistant=is_ai_assistant,
        industry=industry,
        topic=topic,
    )
