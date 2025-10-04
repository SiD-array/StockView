import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAnalytics } from "firebase/analytics";

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyB2t08iUxrAvmeIbz-yoJjoshYQl3Fq3C8",
  authDomain: "stockview-dashboard.firebaseapp.com",
  projectId: "stockview-dashboard",
  storageBucket: "stockview-dashboard.firebasestorage.app",
  messagingSenderId: "711646494621",
  appId: "1:711646494621:web:40112f6e13b44c91d7f974",
  measurementId: "G-JW300NTRFL"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
export const db = getFirestore(app);